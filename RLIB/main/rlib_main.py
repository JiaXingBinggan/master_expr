import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import random
import RLIB.models.RL_brain as rlib
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

def get_model(action_nums, lr, weight_decay, device):
    RL_model = rlib.PolicyGradient(action_nums=action_nums,
            weight_decay=weight_decay,
            learning_rate=lr,
            device=device)

    return RL_model

def get_dataset(args):
    data_path = args.data_path + args.dataset_name + args.campaign_id

    # clk,ctr,mprice,hour,time_frac
    columns = ['clk', 'ctr', 'mprice', 'hour', 'time_frac']
    train_data = pd.read_csv(data_path + 'train.bid.' + args.sample_type + '.data')[columns]
    test_data = pd.read_csv(data_path + 'test.bid.' + args.sample_type + '.data')[columns]

    train_ctr_price = train_data[['mprice', 'ctr']].values.astype(float)
    ascend_train_pctr_price = train_ctr_price[(-train_ctr_price[:, 1]).argsort()]

    auc_ctr_threshold = 0
    expect_auc_num = 0
    print('calculating threshold....\n')

    budget_para = args.budget_para
    for i in range(len(budget_para)):
        first, last = 0, len(ascend_train_pctr_price) - 1
        tmp_budget = args.budget * budget_para[i]
        while first <= last:
            mid = first + (last - first) // 2
            sum_price = np.sum(ascend_train_pctr_price[:mid, 0])
            if sum_price < tmp_budget:
                first = mid + 1
            else:
                if np.sum(ascend_train_pctr_price[:mid - 1, 0]) < tmp_budget:
                    expect_auc_num = mid
                    break
                else:
                    last = mid - 1

        auc_ctr_threshold = ascend_train_pctr_price[expect_auc_num - 1, 1]

    train_data = train_data[['mprice', 'ctr', 'clk', 'hour']].values.astype(float)
    test_data = test_data[['mprice', 'ctr', 'clk', 'hour']].values.astype(float)

    print(len(train_data), len(test_data))
    print(np.sum(train_data[:, 0]), np.sum(test_data[:, 0]))
    print(np.sum(train_data[:, 2]), np.sum(test_data[:, 2]))

    ecpc = np.sum(train_data[:, 0]) / np.sum(train_data[:, 2])
    orgin_ctr = np.sum(train_data[:, 2])/ len(train_data)

    return train_data, test_data, auc_ctr_threshold, expect_auc_num, ecpc, orgin_ctr

def get_list_data(inputs, batch_size, shuffle):
    '''
    :param inputs: List type
    :param batch_size:
    :param shuffle:
    :return:
    '''
    if shuffle:
        np.random.shuffle(inputs)

    while True:
        batch_inputs = inputs[:batch_size, :]
        inputs = inputs[batch_size:, :]

        yield batch_inputs

def get_next_batch(batch):
    return batch.__next__()

def reward_func(bid_price, mprice, win_clk_rate, win_no_clk_rate, remain_budget_on_hour_rate):
    # print(50000 * ctr, mprice)
    if bid_price >= mprice:
        if clk:
            # print('1', win_clk_rate / win_no_clk_rate)
            return (ecpc * ctr - mprice) * (win_clk_rate / win_no_clk_rate)
        else:
            # print('2', - win_no_clk_rate / remain_budget_on_hour_rate if remain_budget_on_hour_rate else 10)
            return - mprice / remain_budget_on_hour_rate if remain_budget_on_hour_rate else mprice
    else:
        if clk:
            # print('3', - (1 - win_clk_rate) * win_no_clk_rate if win_no_clk_rate else 10)
            return - (ecpc * ctr - mprice) * (1 - win_clk_rate)
        else:
            # print('4', (1 - win_no_clk_rate) * remain_budget_on_hour_rate)
            return mprice / win_no_clk_rate if win_no_clk_rate else 10

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
    campaign_id = '1458/'  # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)

    train_data, test_data, auc_ctr_threshold, expect_auc_num, ecpc, origin_ctr \
        = get_dataset(args)

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

    data_clk_index, data_ctr_index, data_mprice_index = args.data_clk_index, args.data_ctr_index, args.data_mprice_index

    device = torch.device(args.device)  # 指定运行设备

    logger.info(campaign_id)
    logger.info('RL model ' + args.model_name + ' has been training')

    actions = np.array(list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) + list(np.arange(100, 301, 10)))

    rl_model = get_model(actions.shape[0], args.lr, args.weight_decay, device)
    B = args.budget * args.budget_para[0]

    train_losses = []

    expect_auc_num = len(train_data)
    logger.info('para:{}, budget:{}'.format(args.budget_para[0], B))
    logger.info('\tclks\treal_clks\tbids\timps\tcost')
    done = 0
    epsilon_max, epsilon_min = 0.9, 0.1
    start_time = datetime.datetime.now()
    stand_scaler = pre.StandardScaler()
    for ep in range(args.episodes):
        budget = B
        next_statistics = [0, 0, 0]  # 用于记录临时特征remain_b,remain_t,next_ctr
        cost_on_no_clk_win_imp, no_clk_win_imps, with_clk_no_win_imps, with_clk_imps, no_clk_no_win_imps, no_clk_imps \
            = 0, 0, 0, 0, 0, 0
        clks, real_clks, bids, imps, cost = 0, 0, 0, 0, 0
        win_clks, win_no_clks, with_clks, with_no_clks = 0, 0, 0, 0

        total_loss = 0
        epsilon = epsilon_min + ep * (epsilon_max - epsilon_min) / args.episodes
        for _, t in enumerate(range(len(train_data))):
            data = train_data[t, :]
            mprice, ctr, clk, hour = data[data_mprice_index], data[data_ctr_index], data[data_clk_index], int(
                data[data_clk_index + 1])

            if hour <= 23 and budget - mprice >= 0:
                real_clks += clk

                if clk:
                    with_clks += 1
                else:
                    with_no_clks += 1

                win_clk_rate = win_clks / with_clks if real_clks else 0
                win_no_clk_rate = win_no_clks / with_no_clks if win_no_clks else 0
                remain_budget_on_hour_rate = (budget / B) / ((23 - hour) / 23)

                s_array = stand_scaler.fit_transform(np.array([remain_budget_on_hour_rate, ctr,
                                                win_clk_rate, win_no_clk_rate]).reshape(-1, 1))
                s_t = torch.Tensor(s_array.flatten())

                bids += 1
                action = rl_model.choose_action(s_t.unsqueeze(0).to(device))
                bid_price = int(actions[action] * ctr / origin_ctr)
                bid_price = bid_price if bid_price <= 300 else 300

                if bid_price >= mprice:
                    cost += mprice
                    clks += clk
                    imps += 1
                    budget -= mprice

                    if clk:
                        win_clks += 1
                    else:
                        win_no_clks += 1

                r_t = reward_func(bid_price, mprice, win_clk_rate, win_no_clk_rate, remain_budget_on_hour_rate)

                rl_model.store_transition(s_t.detach().numpy().tolist(), action, r_t)


        logger.info('ep\t{}\t{}\t{}\t{}\t{}\t{}\t{}s'.format(ep, clks, real_clks, bids, imps, cost,
                                                            [(datetime.datetime.now() - start_time).seconds]))

        loss = rl_model.learn()
        print(loss)
        train_losses.append(total_loss)

    train_loss_df = pd.DataFrame(data=train_losses)
    train_loss_df.to_csv(submission_path + 'train_critics_' + args.sample_type + str(args.budget_para[0]) + '.csv')

    budget = args.budget * args.budget_para[0]
    hour_clks, hour_real_clks, hour_bids, hour_imps, hour_cost = \
        list([0 for _ in range(24)]), list([0 for _ in range(24)]), list([0 for _ in range(24)]), \
        list([0 for _ in range(24)]), list([0 for _ in range(24)])

    cost_on_no_clk_win_imp, no_clk_win_imps, with_clk_no_win_imps, with_clk_imps, no_clk_no_win_imps, no_clk_imps \
        = 0, 0, 0, 0, 0, 0
    clks, real_clks, bids, imps, cost = 0, 0, 0, 0, 0
    bid_datas = []
    with_clks, with_no_clks = 0, 0
    win_clks, win_no_clks = 0, 0
    for t in enumerate(tqdm.tqdm(range(len(test_data)), smoothing=0.0, mininterval=1.0)):
        data = test_data[t, :][0]
        mprice, ctr, clk, hour = data[data_mprice_index], data[data_ctr_index].item(), data[data_clk_index], int(
            data[data_clk_index + 1])

        if hour <= 23 and budget - mprice >= 0:
            real_clks += clk
            hour_real_clks[hour] += clk

            if clk:
                with_clks += 1
            else:
                with_no_clks += 1

            win_clk_rate = win_clks / with_clks if real_clks else 0
            win_no_clk_rate = win_no_clks / with_no_clks if win_no_clks else 0

            remain_budget_on_hour_rate = (budget / B) / ((23 - hour) / 23)

            s_array = stand_scaler.fit_transform(np.array([remain_budget_on_hour_rate, ctr,
                                                           win_clk_rate, win_no_clk_rate]).reshape(-1, 1))
            s_t = torch.Tensor(s_array.flatten())

            bids += 1
            hour_bids[hour] += 1

            action = rl_model.choose_best_action(s_t.unsqueeze(0).to(device))
            bid_price = int(actions[action] * ctr / origin_ctr)
            bid_price = bid_price if bid_price <= 300 else 300

            bid_datas.append(bid_price)

            if bid_price >= mprice:
                cost += mprice
                hour_cost[hour] += mprice

                clks += clk
                hour_clks[hour] += clk

                imps += 1
                hour_imps[hour] += 1

                budget -= mprice

                if clk:
                    win_clks += 1
                else:
                    win_no_clks += 1
        else:
            bid_datas.append(0)
    logger.info('test\t{}\t{}\t{}\t{}\t{}'.format(clks, real_clks, bids, imps, cost))

    hour_records = {'clks': hour_clks, 'real_clks': hour_real_clks, 'bids': hour_bids, 'imps': hour_imps,
                    'cost': hour_cost}
    hour_record_df = pd.DataFrame(data=hour_records)
    hour_record_df.to_csv(submission_path + 'hour_record_' + args.model_name + '_' + args.sample_type +
                          str(args.budget_para[0]) + '.csv')

    record_df = pd.DataFrame(data=[clks, real_clks, bids, imps, cost],
                             index=['clk', 'real_clk', 'bids', 'imps', 'cost'])
    record_df.to_csv(submission_path + 'record_' + args.model_name + '_' + args.sample_type +
                     str(args.budget_para[0]) + '.csv')

    bid_price_df = pd.read_csv(submission_path + 'bid_price_record_' + args.sample_type +
                               str(args.budget_para[0]) + '.csv')
    bid_price_df[args.model_name] = bid_datas
    bid_price_df.to_csv(submission_path + 'bid_price_record_' + args.sample_type +
                        str(args.budget_para[0]) + '.csv')










