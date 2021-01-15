import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import random
import RLIB.models.RL_brain as rlib
import RLIB.models.create_data as Data
import RLIB.models.p_model as Model

import torch
import torch.nn as nn
import torch.utils.data

from itertools import islice
from RLIB.config import config
import logging
import sys
from scipy.optimize import curve_fit

import math

np.seterr(all='raise')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_ctr_model(model_name, feature_nums, field_nums, latent_dims):
    if model_name == 'LR':
        return Model.LR(feature_nums)
    elif model_name == 'FM':
        return Model.FM(feature_nums, latent_dims)
    elif model_name == 'FFM':
        return Model.FFM(feature_nums, field_nums, latent_dims)
    elif model_name == 'W&D':
        return Model.WideAndDeep(feature_nums, field_nums, latent_dims)
    elif model_name == 'DeepFM':
        return Model.DeepFM(feature_nums, field_nums, latent_dims)
    elif model_name == 'FNN':
        return Model.FNN(feature_nums, field_nums, latent_dims)
    elif model_name == 'IPNN':
        return Model.InnerPNN(feature_nums, field_nums, latent_dims)
    elif model_name == 'OPNN':
        return Model.OuterPNN(feature_nums, field_nums, latent_dims)
    elif model_name == 'DCN':
        return Model.DCN(feature_nums, field_nums, latent_dims)
    elif model_name == 'AFM':
        return Model.AFM(feature_nums, field_nums, latent_dims)


def get_model(action_nums, field_nums, latent_dims, lr, weight_decay, device):
    RL_model = rlib.PolicyGradient(action_nums=action_nums,
                                   field_nums=field_nums,
                                   latent_dims=latent_dims,
                                   weight_decay=weight_decay,
                                   learning_rate=lr,
                                   device=device)

    return RL_model


def map_fm(line):
    return line.strip().split(',')


def map_mprice(line):
    return line.strip().split(',')[23]


def bidding(bid):
    return int(bid if bid <= 300 else 300)


def win_rate_f(bid, c):
    win_rate = bid / (c + bid)
    return win_rate


def fit_c(train_data):
    train_data_length = len(train_data)
    prices = list(range(0, 301))
    y_data = list(map(lambda price: len(train_data[price >= train_data[:, 2]]) / train_data_length, prices))
    popt, pcov = curve_fit(win_rate_f, prices, y_data)
    return popt[0]


def generate_bid_price(datas):
    '''
    :param datas: type list
    :return:
    '''
    return np.array(list(map(bidding, datas))).astype(int)


def bid_main(bid_prices, imp_datas, budget):
    '''
    主要竞标程序
    :param bid_prices:
    :param imp_datas:
    :return:
    '''
    win_imp_indexs = np.where(bid_prices >= imp_datas[:, 2])[0]

    win_imp_datas = imp_datas[win_imp_indexs, :]

    win_clks, real_clks, bids, imps, cost = 0, 0, 0, 0, 0
    if len(win_imp_datas):
        first, last = 0, win_imp_datas.shape[0] - 1

        final_index = 0
        while first <= last:
            mid = first + (last - first) // 2
            tmp_sum = np.sum(win_imp_datas[:mid, 2])
            if tmp_sum < budget:
                first = mid + 1
            else:
                last_sum = np.sum(win_imp_datas[:mid - 1, 2])
                if last_sum <= budget:
                    final_index = mid - 1
                    break
                else:
                    last = mid - 1
        final_index = final_index if final_index else first
        win_clks = np.sum(win_imp_datas[:final_index, 0])
        origin_index = win_imp_indexs[final_index - 1]

        real_clks = np.sum(imp_datas[:origin_index, 0])
        imps = final_index + 1
        bids = origin_index + 1

        cost = np.sum(win_imp_datas[:final_index, 2])
        current_cost = np.sum(win_imp_datas[:final_index, 2])

        if len(win_imp_datas[final_index:, :]) > 0:
            if current_cost < budget:
                budget -= current_cost

                final_imps = win_imp_datas[final_index:, :]
                lt_budget_indexs = np.where(final_imps[:, 2] <= budget)[0]

                final_mprice_lt_budget_imps = final_imps[lt_budget_indexs]
                last_win_index = 0
                for idx, imp in enumerate(final_mprice_lt_budget_imps):
                    tmp_mprice = final_mprice_lt_budget_imps[idx, 2]
                    if budget - tmp_mprice >= 0:
                        win_clks += final_mprice_lt_budget_imps[idx, 0]
                        imps += 1
                        bids += (lt_budget_indexs[idx] - last_win_index + 1)
                        last_win_index = lt_budget_indexs[idx]
                        cost += tmp_mprice
                        budget -= tmp_mprice
            else:
                win_clks, real_clks, bids, imps, cost = 0, 0, 0, 0, 0
                last_win_index = 0
                for idx, imp in enumerate(win_imp_datas):
                    tmp_mprice = win_imp_datas[idx, 2]
                    real_clks += win_imp_datas[idx, 0]
                    if budget - tmp_mprice >= 0:
                        win_clks += win_imp_datas[idx, 0]
                        imps += 1
                        bids += (win_imp_indexs[idx] - last_win_index + 1)
                        last_win_index = win_imp_indexs[idx]
                        cost += tmp_mprice
                        budget -= tmp_mprice
    return win_clks, real_clks, bids, imps, cost


def get_dataset(args):
    data_path = args.data_path + args.dataset_name + args.campaign_id

    # clk,ctr,mprice,hour,time_frac
    columns = ['clk', 'ctr', 'mprice', 'hour', 'time_frac']
    train_data = pd.read_csv(data_path + 'train.bid.' + args.sample_type + '.data')[columns]
    test_data = pd.read_csv(data_path + 'test.bid.' + args.sample_type + '.data')[columns]

    train_data = train_data[['clk', 'ctr', 'mprice', 'hour']].values.astype(float)
    test_data = test_data[['clk', 'ctr', 'mprice', 'hour']].values.astype(float)

    ecpc = np.sum(train_data[:, 0]) / np.sum(train_data[:, 2])
    origin_ctr = np.sum(train_data[:, 0]) / len(train_data)
    avg_mprice = np.sum(train_data[:, 2]) / len(train_data)

    return train_data, test_data, ecpc, origin_ctr, avg_mprice


if __name__ == '__main__':
    campaign_id = '3427/'  # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)

    train_data, test_data, ecpc, origin_ctr, avg_mprice = get_dataset(args)

    setup_seed(args.seed)

    # 设置随机数种子
    setup_seed(args.seed)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.save_log_dir + str(args.campaign_id).strip('/') + args.model_name + 'base_output.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    if not os.path.exists(args.save_param_dir + args.campaign_id):
        os.mkdir(args.save_param_dir + args.campaign_id)

    submission_path = args.data_path + args.dataset_name + args.campaign_id + args.model_name + '/'  # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    data_clk_index, data_mprice_index = args.data_clk_index, args.data_mprice_index

    base_algos = ['const', 'rand', 'lin', 'hb', 'ortb']

    paras = list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) + list(np.arange(100, 301, 10))

    # 一定要注意如果顺序的到快花费完预算的时候,例如预算200,但是花到180时,然后下一个曝光的市场价为21,应该去竞标下一个曝光(因为没有那么多钱)

    for budget_para in args.budget_para:
        budget = args.budget * budget_para
        logger.info('Budget para:{}, budget: {}'.format(budget_para, budget))
        # for training
        train_base = {}
        for algo in base_algos:
            clk_dict = {}
            if algo == 'const':
                for para in paras:
                    bid_datas = generate_bid_price(para * np.ones_like(train_data[:, 2]))
                    res_ = bid_main(bid_datas, train_data, budget)
                    clk_dict.setdefault(para, res_[0])
            elif algo == 'rand':
                for para in paras:
                    bid_datas = generate_bid_price(np.random.uniform(0, 1, size=[len(train_data), 1]) * para)
                    res_ = bid_main(bid_datas, train_data, budget)
                    clk_dict.setdefault(para, res_[0])
            elif algo == 'hb':
                for para in paras:
                    bid_datas = generate_bid_price(train_data[:, 1] * para / origin_ctr)
                    res_ = bid_main(bid_datas, train_data, budget)
                    clk_dict.setdefault(para, res_[0])

            clk_dict = sorted(clk_dict.items(), key=lambda x: x[1])
            if clk_dict:
                train_base.setdefault(algo, clk_dict[-1][0])

        logger.info('\t\tclks\treal_clks\tbids\timps\tcost\tpara')
        final_res = {}
        # for testing
        final_bid_datas = {}
        final_bid_datas.setdefault('hour', test_data[:, 3].tolist())
        final_bid_datas.setdefault('ctr', test_data[:, 1].tolist())
        final_bid_datas.setdefault('mprice', test_data[:, 2].tolist())
        final_bid_datas.setdefault('clk', test_data[:, 0].tolist())

        records = []
        bid_datas = np.zeros_like(test_data[:, 2])
        for algo in base_algos:
            if algo == 'const':
                para = train_base[algo]
                bid_datas = generate_bid_price(para * np.ones_like(test_data[:, 2]))
            elif algo == 'rand':
                para = train_base[algo]
                bid_datas = generate_bid_price(np.random.uniform(0, 1, size=[len(test_data), 1]) * para)
            # elif algo == 'mcpc':
            #     bid_datas = ecpc * test_data[:, 1]
            elif algo == 'lin':
                bid_datas = generate_bid_price(test_data[:, 1] * avg_mprice / origin_ctr)
            elif algo == 'hb':
                para = train_base[algo]
                bid_datas = generate_bid_price(test_data[:, 1] * para / origin_ctr)
            elif algo == 'ortb':
                c = fit_c(test_data)
                lamda = 1e-5 # 1.0 5e-6 0.5 1e-5
                bid_datas = generate_bid_price(
                    np.sqrt((test_data[:, 1] * c / lamda) + c * np.ones_like(test_data[:, 1]) ** 2))
            res_ = bid_main(bid_datas, test_data, budget)

            logger.info('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(algo, res_[0], res_[1], res_[2], res_[3], res_[4],
                                                            train_base[algo] if train_base.get(algo) else 0))

            final_bid_datas.setdefault(algo, bid_datas.flatten().tolist())
            records.append([algo, res_[0], res_[1], res_[2], res_[3], res_[4],
                                                            train_base[algo] if train_base.get(algo) else 0])
        bid_price_record_df = pd.DataFrame(data=final_bid_datas)
        bid_price_record_df.to_csv(submission_path + 'bid_price_record_' + args.sample_type + str(budget_para) + '.csv', index = None)

        record = pd.DataFrame(data=records,
                           columns=['algo', 'clks', 'real_clks', 'bids', 'imps', 'cost', 'para'])
        record.to_csv(submission_path + 'record_base_' + args.sample_type + str(budget_para) + '.csv', index=None)





