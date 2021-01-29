import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import random
import DRLB.models.RL_brain_drlb as DRLB
import DRLB.models.reward_net as RewardNet

import torch
import torch.utils.data

from DRLB.config import config
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

    win_clks, real_clks, bids, imps, cost, revenue = 0, 0, 0, 0, 0, 0
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
        revenue = np.sum(win_imp_datas[:final_index, 1])
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
                        revenue += final_mprice_lt_budget_imps[idx, 1]
                        last_win_index = lt_budget_indexs[idx]
                        cost += tmp_mprice
                        budget -= tmp_mprice
                    else:
                        break
                real_clks += np.sum(final_imps[:last_win_index, 0])
            else:
                win_clks, real_clks, bids, imps, cost, revenue = 0, 0, 0, 0, 0, 0
                last_win_index = 0
                for idx, imp in enumerate(win_imp_datas):
                    tmp_mprice = win_imp_datas[idx, 2]
                    real_clks += win_imp_datas[idx, 0]
                    if budget - tmp_mprice >= 0:
                        win_clks += win_imp_datas[idx, 0]
                        imps += 1
                        bids += (win_imp_indexs[idx] - last_win_index + 1)
                        revenue += win_imp_datas[idx, 1]
                        last_win_index = win_imp_indexs[idx]
                        cost += tmp_mprice
                        budget -= tmp_mprice

    return win_clks, real_clks, bids, imps, cost, revenue


def get_model(args, device):
    action_space = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
    RL_model = DRLB.DRLB(args.neuron_nums,
                         action_dims=len(action_space),
                         state_dims=7,
                         lr=args.lr,
                         memory_size=args.memory_size,
                         batch_size=args.rl_batch_size,
                         device=device)

    Reward_Net_model = RewardNet.RewardNet(args.neuron_nums,
                                           action_dims=1,
                                           reward_dims=1,
                                           state_dims=7,
                                           lr=args.lr,
                                           memory_size=args.memory_size,
                                           batch_size=args.rl_batch_size,
                                           device=device)

    return RL_model, Reward_Net_model, action_space


def get_dataset(args):
    data_path = args.data_path + args.dataset_name + args.campaign_id

    # clk,ctr,mprice,hour,time_frac
    columns = ['clk', 'ctr', 'mprice', 'hour', 'time_frac']
    train_data = pd.read_csv(data_path + 'train.bid.' + args.sample_type + '.data')[columns]
    test_data = pd.read_csv(data_path + 'test.bid.' + args.sample_type + '.data')[columns]

    train_data = train_data[['clk', 'ctr', 'mprice', 'time_frac']].values.astype(float)
    test_data = test_data[['clk', 'ctr', 'mprice', 'time_frac']].values.astype(float)

    ecpc = np.sum(train_data[:, 0]) / np.sum(train_data[:, 2])
    origin_ctr = np.sum(train_data[:, 0]) / len(train_data)

    return train_data, test_data, ecpc, origin_ctr


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

    train_data, test_data, ecpc, origin_ctr = get_dataset(args)

    setup_seed(args.seed)

    log_dirs = [args.save_log_dir, args.save_log_dir + args.campaign_id]
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    param_dirs = [args.save_param_dir, args.save_param_dir + args.campaign_id]
    for param_dir in param_dirs:
        if not os.path.exists(param_dir):
            os.mkdir(param_dir)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.save_log_dir + str(args.campaign_id).strip('/') + args.model_name + '_output.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    submission_path = args.data_path + args.dataset_name + args.campaign_id + args.model_name + '/'  # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    device = torch.device(args.device)  # 指定运行设备

    logger.info(campaign_id)
    logger.info('RL model ' + args.model_name + ' has been training')
    logger.info(args)

    actions = np.array(list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) + list(np.arange(100, 301, 10)))

    rl_model, reward_net_model, action_space = get_model(args, device)

    B = args.budget * args.budget_para[0]

    hb_clk_dict = {}
    for para in actions:
        bid_datas = generate_bid_price(train_data[:, 1] * para / origin_ctr)
        res_ = bid_main(bid_datas, train_data, B)
        hb_clk_dict.setdefault(para, res_[0])

    hb_base = sorted(hb_clk_dict.items(), key=lambda x: x[1])[-1][0]

    Lamda = hb_base / origin_ctr

    train_losses = []

    logger.info('para:{}, budget:{}, base bid: {}'.format(args.budget_para[0], B, hb_base))
    logger.info('\tclks\treal_clks\tbids\timps\tcost')

    start_time = datetime.datetime.now()

    clk_index, ctr_index, mprice_index, hour_index = 0, 1, 2, 3

    ep_train_records = []
    ep_test_records = []
    ep_test_actions = []

    rl_model.reset_epsilon(0.9)  # init epsilon value

    for ep in range(args.episodes):
        if ep % 10 == 0:
            budget = B

            init_lamda = Lamda
            tmp_lamda = 0

            tmp_state = [1, 1, 95, 0, 0, 0, 0]
            init_state = [1, 1, 95, 0, 0, 0, 0]

            tmp_budget = 0

            test_records = [0, 0, 0, 0, 0, 0]
            test_actions = [0 for _ in range(96)]
            for t in range(96):
                if budget > 0:
                    hour_datas = test_data[test_data[:, hour_index] == t]

                    state = torch.tensor(init_state).float() if not t else torch.tensor(tmp_state).float()
                    lamda = init_lamda if not t else tmp_lamda

                    action = rl_model.choose_action(state.unsqueeze(0).to(device))

                    test_actions[t] = action_space[action]

                    lamda = lamda * (1 + action_space[action])
                    tmp_lamda = lamda

                    bid_datas = generate_bid_price((hour_datas[:, ctr_index] * lamda))
                    res_ = bid_main(bid_datas, hour_datas, budget)
                    # win_clks, real_clks, bids, imps, cost, revenue

                    test_records = [test_records[i] + res_[i] for i in range(len(test_records))]
                    tmp_budget = budget
                    budget -= res_[4]

                    # state: time_steps, remain_b, left regulation times for lambda, (b_t - b_t-1) / b_t-1, cpm_t, wr_t,
                    # win_pctrs_t( sum(x_i * v_i), x_i denotes whether win imp, v_i denots the imp's pctr)
                    next_state = [(t + 2), budget / B, (96 - (t + 2)),
                                  (budget - tmp_budget) / tmp_budget if tmp_budget else 0,
                                  res_[4] / res_[3] if res_[3] else 0,
                                  res_[3] / res_[2] if res_[2] else 0,
                                  res_[5]]
                    tmp_state = next_state
            ep_test_records.append([ep] + test_records)
            ep_test_actions.append(test_actions)
            print(ep, 'test', test_records)

        budget = B

        init_lamda = Lamda
        tmp_lamda = 0

        tmp_state = [1, 1, 95, 0, 0, 0, 0]
        init_state = [1, 1, 95, 0, 0, 0, 0]

        tmp_budget = 0

        train_records = [0, 0, 0, 0, 0, 0]
        # state: time_steps, remain_b, left regulation times for lambda, (b_t - b_t-1) / b_t-1, cpm_t, wr_t,
        # win_pctrs_t( sum(x_i * v_i), x_i denotes whether win imp, v_i denots the imp's pctr)
        critic_loss = 0

        done = 0
        state_action_pairs = []
        V = 0

        for t in range(96):
            if budget > 0:
                if reward_net_model.memory_D_counter >= args.rl_batch_size:
                    reward_net_model.learn()

                hour_datas = train_data[train_data[:, hour_index] == t]

                state = torch.tensor(init_state).float() if not t else torch.tensor(tmp_state).float()
                lamda = init_lamda if not t else tmp_lamda

                action = rl_model.choose_action(state.unsqueeze(0).to(device))

                lamda = lamda * (1 + action_space[action])
                tmp_lamda = lamda

                # Lamda = Lamda * (1 + action_space[action])

                bid_datas = generate_bid_price((hour_datas[:, ctr_index] * lamda))
                res_ = bid_main(bid_datas, hour_datas, budget)
                # win_clks, real_clks, bids, imps, cost, revenue

                train_records = [train_records[i] + res_[i] for i in range(len(train_records))]
                tmp_budget = budget
                budget -= res_[4]

                left_frac_ratio = (96 - (t + 1))
                if (not left_frac_ratio) or (budget <= 0):
                    done = 1

                # state: time_steps, remain_b, left regulation times for lambda, (b_t - b_t-1) / b_t-1, cpm_t, wr_t,
                # win_pctrs_t( sum(x_i * v_i), x_i denotes whether win imp, v_i denots the imp's pctr)
                next_state = [(t + 2), budget / B, (96 - (t + 2)),
                              (budget - tmp_budget) / tmp_budget if tmp_budget else 0,
                              res_[4] / res_[3] if res_[3] else 0,
                              res_[3] / res_[2] if res_[2] else 0,
                              res_[5]]
                tmp_state = next_state

                state_action = torch.cat([state, torch.tensor([action_space[action]]).float()], dim=-1).unsqueeze(0).to(device)
                r_t = reward_net_model.return_model_reward(state_action)

                transitions = torch.cat([state, torch.tensor([action]).float(),
                                         torch.tensor([r_t]).float(), torch.tensor(next_state).float(),
                                         torch.tensor([done]).float()], dim=-1).unsqueeze(
                    0).to(device)

                rl_model.store_transition(transitions)

                state_action_pairs.append((state, action_space[action]))

                V += res_[5]

                rl_model.control_epsilon(ep * (t + 1))

                if rl_model.memory_counter >= args.rl_batch_size:
                    critic_loss = rl_model.learn()

        for (s, a) in state_action_pairs:
            state_action = tuple(np.append(s, a))
            max_rtn = max(reward_net_model.get_reward_from_S(state_action), V)
            reward_net_model.store_S_pair(state_action, max_rtn)
            reward_net_model.store_D_pair(torch.cat([s, torch.tensor([a]).float(), torch.tensor([max_rtn]).float()], dim=-1))

        if ep % 10 == 0:
            ep_train_records.append([ep] + train_records + [critic_loss])

    train_record_df = pd.DataFrame(data=ep_train_records,
                                   columns=['ep', 'clks', 'real_clks', 'bids', 'imps', 'cost', 'revenue', 'loss'])
    train_record_df.to_csv(submission_path + 'drlb_train_records_' + str(args.budget_para[0]) + '.csv', index=None)

    test_record_df = pd.DataFrame(data=ep_test_records,
                                  columns=['ep', 'clks', 'real_clks', 'bids', 'imps', 'cost', 'revenue'])
    test_record_df.to_csv(submission_path + 'drlb_test_records_' + str(args.budget_para[0]) + '.csv', index=None)

    test_action_df = pd.DataFrame(data=ep_test_actions)
    test_action_df.to_csv(submission_path + 'drlb_test_actions_' + str(args.budget_para[0]) + '.csv')