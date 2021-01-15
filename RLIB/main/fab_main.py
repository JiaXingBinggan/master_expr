import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import random
import RLIB.models.RL_bain_fab as td3
import RLIB.models.create_data as Data
import RLIB.models.p_model as Model
from RLIB.models.Feature_embedding import Feature_Embedding

import sklearn.preprocessing as pre

import tqdm

import torch
import torch.nn as nn
import torch.utils.data

from itertools import islice
from RLIB.config import config
import logging
import sys

np.seterr(all='raise')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def bidding(bid):
    return int(bid if bid <= 300 else 300)

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


def get_model(batch_size, device):
    RL_model = td3.TD3_Model(action_nums=1,
                             batch_size=batch_size,
                             device=device
                             )

    return RL_model


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
    print(origin_ctr)
    return train_data, test_data, ecpc, origin_ctr, avg_mprice

def reward_func(fab_clks, hb_clks, fab_cost, hb_cost):
    if fab_clks >= hb_clks and fab_cost < hb_cost:
        r = 5
    elif fab_clks >= hb_clks and fab_cost >= hb_cost:
        r = 1
    elif fab_clks < hb_clks and fab_cost >= hb_cost:
        r = -5
    else:
        r = -2.5

    return r / 10000

'''
1458
437520 447493
30309883.0 30297100.0
395.0 356.0

3358
237844 335310
23340047.0 32515709.0
197.0 307.0

3386
412275 392901
32967478.0 31379459.0
344.0 355.0

3427
379524 390398
30918866.0 31654042.0
282.0 313.0

'''

if __name__ == '__main__':
    campaign_id = '3427/'  # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)

    train_data, test_data, ecpc, origin_ctr, avg_mprice = get_dataset(args)

    args.model_name= 'FAB'

    setup_seed(args.seed)

    # 设置随机数种子
    setup_seed(args.seed)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.save_log_dir + str(args.campaign_id).strip('/') + args.model_name + '_output.log',
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

    device = torch.device(args.device)  # 指定运行设备

    logger.info(campaign_id)
    logger.info('RL model ' + args.model_name + ' has been training')

    actions = np.array(list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) + list(np.arange(100, 301, 10)))

    rl_model = get_model(args.rl_batch_size, device)
    B = args.budget * args.budget_para[0]
    print(B)
    #
    # hb_clk_dict = {}
    # for para in actions:
    #     bid_datas = generate_bid_price(train_data[:, 1] * para / origin_ctr)
    #     res_ = bid_main(bid_datas, train_data, B)
    #     hb_clk_dict.setdefault(para, res_[0])
    #
    # hb_base = sorted(hb_clk_dict.items(), key=lambda x: x[1])[-1][0]
    # print(hb_base)
    hb_base = 60
    hour_data = test_data[test_data[:, 3] == 0]
    bid_datas = generate_bid_price(hour_data[:, 1] * hb_base / origin_ctr)
    res_ = bid_main(bid_datas, hour_data, B)
    print(res_)
    train_losses = []

    logger.info('para:{}, budget:{}, base bid: {}'.format(args.budget_para[0], B, hb_base))
    logger.info('\tclks\treal_clks\tbids\timps\tcost')

    start_time = datetime.datetime.now()

    clk_index, ctr_index, mprice_index, hour_index = 0, 1, 2, 3

    for ep in range(args.episodes):
        budget = B

        tmp_state = [1, 0, 0, 0]
        init_state = [1, 0, 0, 0]
        records = [0, 0, 0, 0, 0]
        # win_clks, real_clks, bids, imps, cost
        # win_clks, real_clks, bids, imps, cost = 0, 0, 0, 0, 0
        critic_loss = 0

        done = 0
        for t in range(24):
            if budget > 0:
                hour_datas = train_data[train_data[:, hour_index] == t]

                state = torch.tensor(init_state).float() if not t else torch.tensor(tmp_state).float()

                action = rl_model.choose_action(state.unsqueeze(0))[0, 0].item()

                bid_datas = generate_bid_price((hour_datas[:, ctr_index] * (hb_base / origin_ctr)) / (1 + action))
                res_ = bid_main(bid_datas, hour_datas, budget)
                # win_clks, real_clks, bids, imps, cost

                records = [records[i] + res_[i] for i in range(len(records))]
                budget -= res_[-1]

                left_hour_ratio = (23 - t) / 23 if t <= 23 else 0
                if left_hour_ratio:
                    # avg_budget_ratio, cost_ratio, ctr, win_rate
                    next_state = [(budget / B) / left_hour_ratio if left_hour_ratio else 0,
                                  res_[4] / B,
                                  res_[0] / res_[3] if res_[3] else 0,
                                  res_[3] / res_[2] if res_[2] else 0]
                    tmp_state = next_state
                else:
                    next_state = [0,
                                  res_[4] / B,
                                  res_[0] / res_[3] if res_[3] else 0,
                                  res_[3] / res_[2] if res_[2] else 0]
                    tmp_state = next_state
                    done = 1

                hb_bid_datas = generate_bid_price(hour_datas[:, ctr_index] * hb_base / origin_ctr)
                res_hb = bid_main(hb_bid_datas, hour_datas, budget)

                r_t = reward_func(res_[0], res_hb[0], res_[3], res_hb[3])

                transitions = torch.cat([state, torch.tensor([action]).float(),
                                         torch.tensor(next_state).float(),
                                         torch.tensor([done]).float(), torch.tensor([r_t]).float()], dim=-1).unsqueeze(0).to(device)

                rl_model.store_transition(transitions)

                if rl_model.memory.memory_counter >= args.rl_batch_size:
                    critic_loss = rl_model.learn()

        # print('train', records, critic_loss)

        test_records = [0, 0, 0, 0, 0]
        tmp_test_state = [1, 0, 0, 0]
        init_test_state = [1, 0, 0, 0]
        test_rewards = 0
        budget = B
        for t in range(24):
            if budget > 0:
                hour_datas = test_data[test_data[:, hour_index] == t]

                state = torch.tensor(init_test_state).float() if not t else torch.tensor(tmp_test_state).float()

                action = rl_model.choose_action(state.unsqueeze(0))[0, 0].item()

                bid_datas = generate_bid_price((hour_datas[:, ctr_index] * hb_base / origin_ctr) / (1 + action))
                res_ = bid_main(bid_datas, hour_datas, budget)

                # win_clks, real_clks, bids, imps, cost

                test_records = [test_records[i] + res_[i] for i in range(len(records))]
                budget -= res_[-1]

                hb_bid_datas = generate_bid_price(hour_datas[:, ctr_index] * hb_base / origin_ctr)
                res_hb = bid_main(hb_bid_datas, hour_datas, budget)

                r_t = reward_func(res_[0], res_hb[0], res_[3], res_hb[3])
                test_rewards += r_t

                left_hour_ratio = (23 - t) / 23 if t <= 23 else 0
                # avg_budget_ratio, cost_ratio, ctr, win_rate
                next_state = [(budget / B) / left_hour_ratio if left_hour_ratio else 0,
                              res_[4] / B,
                              res_[0] / res_[3] if res_[3] else 0,
                              res_[3] / res_[2] if res_[2] else 0]
                tmp_test_state = next_state

        print(ep, 'test', test_records, test_rewards)




