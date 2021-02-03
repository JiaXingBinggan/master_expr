import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
from sklearn.metrics import roc_auc_score
import src.models.p_model as Model
import src.models.v12_Hybrid_TD3_model_PER as td3_model
import src.models.creat_data as Data
from src.models.Feature_embedding import Feature_Embedding

import torch
import torch.nn as nn
import torch.utils.data

from src.config import config
import logging
import sys

np.seterr(all='raise')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(args, device):
    RL_model = td3_model.Hybrid_TD3_Model(neuron_nums=args.neuron_nums,
                                          action_nums=args.ensemble_nums,
                                          lr_C_A=args.init_lr_a,
                                          lr_D_A=args.init_lr_a,
                                          lr_C=args.init_lr_c,
                                          batch_size=args.rl_batch_size,
                                          memory_size=args.memory_size,
                                          random_seed=args.seed,
                                          device=device
                                          )

    return RL_model

def generate_preds(ensemble_nums, pretrain_y_preds, actions, prob_weights, c_actions,
                   labels, device, mode):
    y_preds = torch.ones_like(actions, dtype=torch.float)
    rewards = torch.ones_like(actions, dtype=torch.float)

    sort_prob_weights, sortindex_prob_weights = torch.sort(-prob_weights, dim=1)

    return_c_actions = torch.zeros(size=(pretrain_y_preds.size()[0], ensemble_nums)).to(device)
    return_ctrs = torch.zeros(size=(pretrain_y_preds.size()[0], ensemble_nums)).to(device)

    choose_model_lens = range(1, ensemble_nums + 1)
    for i in choose_model_lens:  # 根据ddqn_model的action,判断要选择ensemble的数量
        with_action_indexs = (actions == i).nonzero()[:, 0]
        current_choose_models = sortindex_prob_weights[with_action_indexs][:, :i]
        current_basic_rewards = torch.ones(size=[len(with_action_indexs), 1]).to(device) * 1
        current_prob_weights = prob_weights[with_action_indexs]

        current_with_clk_indexs = (labels[with_action_indexs] == 1).nonzero()[:, 0]
        current_without_clk_indexs = (labels[with_action_indexs] == 0).nonzero()[:, 0]

        current_pretrain_y_preds = pretrain_y_preds[with_action_indexs, :]

        current_c_actions = c_actions[with_action_indexs, :]
        current_ctrs = current_pretrain_y_preds
        if i == ensemble_nums:
            current_y_preds = torch.sum(torch.mul(current_prob_weights, current_pretrain_y_preds), dim=1).view(-1, 1)
            y_preds[with_action_indexs, :] = current_y_preds

            return_c_actions[with_action_indexs, :] = current_c_actions
            return_ctrs[with_action_indexs, :] = current_ctrs
        elif i == 1:
            current_y_preds = torch.ones(size=[len(with_action_indexs), 1]).to(device)
            current_c_actions_temp = torch.zeros(size=[len(with_action_indexs), ensemble_nums]).to(device)
            current_ctrs_temp = torch.zeros(size=[len(with_action_indexs), ensemble_nums]).to(device)
            for k in range(ensemble_nums):
                choose_model_indexs = (current_choose_models == k).nonzero()[:, 0] # 找出下标
                current_y_preds[choose_model_indexs, 0] = current_pretrain_y_preds[choose_model_indexs, k]

                current_c_actions_temp[choose_model_indexs, k] = current_c_actions[choose_model_indexs, k]

                current_ctrs_temp[choose_model_indexs, k] = current_ctrs[choose_model_indexs, k]

            y_preds[with_action_indexs, :] = current_y_preds
            return_c_actions[with_action_indexs, :] = current_c_actions_temp
            return_ctrs[with_action_indexs, :] = current_ctrs_temp
        else:
            current_softmax_weights = torch.softmax(
                sort_prob_weights[with_action_indexs][:, :i] * -1, dim=1
            ).to(device)  # 再进行softmax

            current_row_preds = torch.ones(size=[len(with_action_indexs), i]).to(device)
            current_c_actions_temp = torch.zeros(size=[len(with_action_indexs), ensemble_nums]).to(device)
            current_ctrs_temp = torch.zeros(size=[len(with_action_indexs), ensemble_nums]).to(device)
            current_softmax_weights_temp = torch.zeros(size=[len(with_action_indexs), ensemble_nums]).to(device)

            for m in range(i):
                current_row_choose_models = current_choose_models[:, m:m + 1]  # 这个和current_c_actions等长

                for k in range(ensemble_nums):
                    current_pretrain_y_pred = current_pretrain_y_preds[:, k: k + 1]
                    choose_model_indexs = (current_row_choose_models == k).nonzero()[:, 0]

                    current_row_preds[choose_model_indexs, m:m + 1] = current_pretrain_y_pred[choose_model_indexs]

                    current_c_actions_temp[choose_model_indexs, k:k + 1] = current_c_actions[choose_model_indexs,
                                                                           k: k + 1]

                    current_ctrs_temp[choose_model_indexs, k: k + 1] = current_ctrs[choose_model_indexs, k: k + 1]

                    current_softmax_weights_temp[choose_model_indexs, k: k + 1] = current_softmax_weights[
                                                                                  choose_model_indexs, m: m + 1]

            current_y_preds = torch.sum(torch.mul(current_softmax_weights, current_row_preds), dim=1).view(-1, 1)
            y_preds[with_action_indexs, :] = current_y_preds

            return_c_actions[with_action_indexs, :] = current_c_actions_temp  # 为了让没有使用到的位置,值置为0
            return_ctrs[with_action_indexs, :] = torch.mul(current_softmax_weights_temp, current_ctrs_temp)

        with_clk_rewards = torch.where(
            current_y_preds[current_with_clk_indexs] > current_pretrain_y_preds[
                current_with_clk_indexs].mean(dim=1).view(-1, 1),
            current_basic_rewards[current_with_clk_indexs] * 1e-1,
            current_basic_rewards[current_with_clk_indexs] * -1e-1
        )

        # with_clk_rewards = torch.log(current_y_preds[current_with_clk_indexs])

        without_clk_rewards = torch.where(
            current_y_preds[current_without_clk_indexs] < current_pretrain_y_preds[
                current_without_clk_indexs].mean(dim=1).view(-1, 1),
            current_basic_rewards[current_without_clk_indexs] * 1e-1,
            current_basic_rewards[current_without_clk_indexs] * -1e-1
        )

        # without_clk_rewards = torch.log(torch.ones_like(labels[current_without_clk_indexs]).float() -
        #                        current_y_preds[[current_without_clk_indexs]])

        current_basic_rewards[current_with_clk_indexs] = with_clk_rewards
        current_basic_rewards[current_without_clk_indexs] = without_clk_rewards

        rewards[with_action_indexs, :] = current_basic_rewards

    return y_preds, rewards, return_c_actions, return_ctrs


def test(rl_model, ensemble_nums, data_loader, device):
    targets, predicts = list(), list()
    intervals = 0
    total_test_loss = 0
    test_rewards = torch.FloatTensor().to(device)
    final_actions = torch.LongTensor().to(device)
    final_prob_weights = torch.FloatTensor().to(device)
    with torch.no_grad():
        for i, (current_pretrain_y_preds, labels) in enumerate(data_loader):
            current_pretrain_y_preds, labels = current_pretrain_y_preds.float().to(device), torch.unsqueeze(labels, 1).to(
                device)

            s_t = torch.cat([current_pretrain_y_preds.mean(dim=-1).view(-1, 1), current_pretrain_y_preds], dim=-1)

            actions, c_actions, ensemble_c_actions = rl_model.choose_best_action(s_t)

            y, rewards, return_c_actions, return_ctrs = generate_preds(ensemble_nums, current_pretrain_y_preds, actions, ensemble_c_actions, c_actions,
                                                          labels, device, mode='test')

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
            intervals += 1

            test_rewards = torch.cat([test_rewards, rewards], dim=0)

            final_actions = torch.cat([final_actions, actions], dim=0)
            final_prob_weights = torch.cat([final_prob_weights, ensemble_c_actions], dim=0)

    return roc_auc_score(targets, predicts), predicts, test_rewards.mean().item(), final_actions, final_prob_weights


def submission(rl_model, ensemble_nums, data_loader, device):
    targets, predicts = list(), list()
    final_actions = torch.LongTensor().to(device)
    final_prob_weights = torch.FloatTensor().to(device)
    with torch.no_grad():
        for i, (current_pretrain_y_preds, labels) in enumerate(data_loader):
            current_pretrain_y_preds, labels = current_pretrain_y_preds.float().to(device), torch.unsqueeze(labels, 1).to(device)

            s_t = torch.cat([current_pretrain_y_preds.mean(dim=-1).view(-1, 1), current_pretrain_y_preds], dim=-1)

            actions, c_actions, ensemble_c_actions = rl_model.choose_best_action(s_t)

            y, rewards, return_c_actions, return_ctrs = generate_preds(ensemble_nums, current_pretrain_y_preds, actions, ensemble_c_actions, c_actions,
                                                          labels, device, mode='test')

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())

            final_actions = torch.cat([final_actions, actions], dim=0)
            final_prob_weights = torch.cat([final_prob_weights, ensemble_c_actions], dim=0)

    return predicts, roc_auc_score(targets, predicts), final_actions.cpu().numpy(), final_prob_weights.cpu().numpy()

def get_ensemble_model(model_name, feature_nums, field_nums, latent_dims):
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
        batch_inputs = inputs[0: batch_size]
        inputs = np.concatenate([inputs[batch_size:], inputs[:batch_size]], axis=0)

        yield batch_inputs

def eva_stopping(valid_rewards, args):  # early stopping
    if len(valid_rewards) >= args.rl_early_stop_iter:

        reward_campare_arrs = [valid_rewards[-i][1] < valid_rewards[-i - 1][1] for i in range(1, args.rl_early_stop_iter)]
        reward_div_mean = sum([abs(valid_rewards[-i][1] - valid_rewards[-i - 1][1]) for i in range(1, args.rl_early_stop_iter)]) / args.rl_early_stop_iter

        if (False not in reward_campare_arrs) or (reward_div_mean <= args.reward_epsilon):
            return True

    return False

def get_dataset(args):
    datapath = args.data_path + args.dataset_name + args.campaign_id

    columns = ['label'] + args.ensemble_models.split(',')
    train_data = pd.read_csv(datapath + 'train.rl_ctr.' + args.sample_type + '.txt')[columns].values.astype(float)
    val_data = pd.read_csv(datapath + 'val.rl_ctr.' + args.sample_type + '.txt')[columns].values.astype(float)
    test_data = pd.read_csv(datapath + 'test.rl_ctr.' + args.sample_type + '.txt')[columns].values.astype(float)

    return train_data, val_data, test_data

if __name__ == '__main__':
    campaign_id = '2259/'  # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)

    train_data, val_data, test_data = get_dataset(args)

    # 设置随机数种子
    setup_seed(args.seed)

    log_dirs = [args.save_log_dir, args.save_log_dir + args.campaign_id]
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.save_log_dir + str(args.campaign_id).strip('/') + args.rl_model_name + '_output.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    if not os.path.exists(args.save_param_dir + args.campaign_id):
        os.mkdir(args.save_param_dir + args.campaign_id)

    submission_path = args.data_path + args.dataset_name + args.campaign_id + args.rl_model_name + '/'  # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    device = torch.device(args.device)  # 指定运行设备

    logger.info(campaign_id)
    logger.info('RL model ' + args.rl_model_name + ' has been training')
    logger.info(args)

    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.rl_gen_batch_size, num_workers=8)

    val_dataset = Data.libsvm_dataset(val_data[:, 1:], val_data[:, 0])
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.rl_gen_batch_size, num_workers=8)

    test_predict_arrs = []

    model_dict_len = args.ensemble_nums

    gap = args.run_steps // args.record_times
    # gap = 10000

    data_len = len(train_data)

    rl_model = get_model(args, device)

    loss = nn.BCELoss()

    val_aucs = []

    val_rewards_records = []
    timesteps = []
    train_critics = []
    global_steps = 0

    early_aucs, early_rewards = [], []

    random = True
    start_time = datetime.datetime.now()

    torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

    train_start_time = datetime.datetime.now()

    is_sample_action = True # 是否在训练开始时完全使用随机动作
    is_val = False

    tmp_train_ctritics = 0

    record_param_steps = 0
    is_early_stop = False
    early_stop_index = 0
    intime_steps = 0

    record_list = {}

    # 设计为每隔rl_iter_size的次数训练以及在测试集上测试一次
    # 总的来说,对于ipinyou,训练集最大308万条曝光,所以就以500万次结果后,选取连续early_stop N 轮(N轮rewards没有太大变化)中auc最高的的模型进行生成
    train_batch_gen = get_list_data(train_data, args.rl_iter_size, True)# 要不要早停
    # train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    # train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.rl_iter_size, num_workers=8, shuffle=1)
    while intime_steps <= args.stop_steps:
        batchs = train_batch_gen.__next__()
        labels = torch.Tensor(batchs[:, 0: 1]).long().to(device)
        current_pretrain_y_preds = torch.Tensor(batchs[:, 1:]).float().to(device)
    # for i, (pretrain_y_preds, labels) in enumerate(train_data_loader):
    #     intime_steps = i * args.rl_iter_size
    #     current_pretrain_y_preds, labels = pretrain_y_preds.float().to(device), labels.long().unsqueeze(1).to(device)

        s_t = torch.cat([current_pretrain_y_preds.mean(dim=-1).view(-1, 1), current_pretrain_y_preds], dim=-1)

        c_actions, ensemble_c_actions, d_q_values, ensemble_d_actions = rl_model.choose_action(
            s_t, False if intime_steps >= gap else True)

        y_preds, rewards, return_c_actions, return_ctrs = \
            generate_preds(args.ensemble_nums, current_pretrain_y_preds, ensemble_d_actions, ensemble_c_actions, c_actions, labels, device,
                           mode='train')

        s_t_ = torch.cat([y_preds, return_ctrs], dim=-1)

        transitions = torch.cat(
            [s_t, return_c_actions, d_q_values, s_t_, rewards],
            dim=1)

        rl_model.store_transition(transitions)

        if intime_steps % gap == 0:
            val_auc, val_predicts, val_rewards, val_actions, val_prob_weights = test(rl_model, args.ensemble_nums, val_data_loader, device)
            auc, predicts, test_rewards, actions, prob_weights = test(rl_model, args.ensemble_nums, test_data_loader, device)

            record_list[np.mod(record_param_steps, args.rl_early_stop_iter)] = [auc, predicts, actions, prob_weights]

            logger.info('Model {}, timesteps {}, val auc {}, val rewards {}, test auc {}, [{}s]'.format(
                args.rl_model_name, intime_steps, val_auc, val_rewards, auc, (datetime.datetime.now() - train_start_time).seconds))
            val_rewards_records.append(val_rewards)
            timesteps.append(intime_steps)
            val_aucs.append(val_auc)

            train_critics.append(tmp_train_ctritics)

            # rl_model.temprature = max(rl_model.temprature_min,
            #                            rl_model.temprature - gap *
            #                            (rl_model.temprature_max - rl_model.temprature_min) / args.run_steps)
            # rl_model.memory.beta = min(rl_model.memory.beta_max,
            #                           rl_model.memory.beta + gap *
            #                            (rl_model.memory.beta_max - rl_model.memory.beta_min) / args.run_steps)

            early_aucs.append([record_param_steps, val_auc])
            early_rewards.append([record_param_steps, test_rewards])
            torch.save(rl_model.Hybrid_Actor.state_dict(),
                       args.save_param_dir + args.campaign_id + args.rl_model_name + str(
                           np.mod(record_param_steps, args.rl_early_stop_iter)) + '.pth')

            record_param_steps += 1
            if args.run_steps <= intime_steps <= args.stop_steps:
                if eva_stopping(early_rewards, args):
                    max_auc_index = sorted(early_aucs[-args.rl_early_stop_iter:], key=lambda x: x[1], reverse=True)[0][
                        0]
                    early_stop_index = np.mod(max_auc_index, args.rl_early_stop_iter)
                    is_early_stop = True
                    break

            torch.cuda.empty_cache()

        if intime_steps >= args.rl_batch_size and intime_steps % 10 == 0:
        # if intime_steps >= args.rl_batch_size:
            critic_loss = rl_model.learn()
            tmp_train_ctritics = critic_loss

        intime_steps += batchs.shape[0]

    logger.info('Final gumbel Softmax temprature is {}'.format(rl_model.temprature))

    train_end_time = datetime.datetime.now()

    '''
    要不要早停
    '''
    # if is_early_stop:
    #     test_rl_model = get_model(args, device)
    #     load_path = args.save_param_dir + args.campaign_id + args.rl_model_name + str(
    #                           early_stop_index) + '.pth'
    #
    #     test_rl_model.Hybrid_Actor.load_state_dict(torch.load(load_path, map_location=device))  # 加载最优参数
    # else:
    #     test_rl_model = rl_model

    # test_predicts, test_auc, test_actions, test_prob_weights = submission(test_rl_model, args.ensemble_nums, test_data_loader,
    #                                                                       device)
    test_auc, test_predicts, test_actions, test_prob_weights = record_list[early_stop_index]

    # for i in range(args.rl_early_stop_iter):
    #     os.remove(args.save_param_dir + args.campaign_id + args.rl_model_name + str(i) + '.pth')

    logger.info('Model {}, test auc {}, [{}s]'.format(args.rl_model_name,
                                                        test_auc, (datetime.datetime.now() - start_time).seconds))
    test_predict_arrs.append(test_predicts)

    neuron_nums_str = '_'.join(map(str, args.neuron_nums))

    prob_weights_df = pd.DataFrame(data=test_prob_weights.cpu().numpy())
    prob_weights_df.to_csv(submission_path + 'test_prob_weights_' + str(args.ensemble_nums) + '_'
                           + args.sample_type + neuron_nums_str + '.csv', index=None)

    actions_df = pd.DataFrame(data=test_actions.cpu().numpy())
    actions_df.to_csv(submission_path + 'test_actions_' + str(args.ensemble_nums) + '_'
                      + args.sample_type + '_' + neuron_nums_str + '.csv', index=None)

    valid_aucs_df = pd.DataFrame(data=val_aucs)
    valid_aucs_df.to_csv(submission_path + 'val_aucs_' + str(args.ensemble_nums) + '_'
                         + args.sample_type + '_' + neuron_nums_str + '.csv', index=None)

    val_rewards_records = {'rewards': val_rewards_records, 'timesteps': timesteps}
    val_rewards_records_df = pd.DataFrame(data=val_rewards_records)
    val_rewards_records_df.to_csv(submission_path + 'val_reward_records_' + str(args.ensemble_nums) + '_'
                                  + args.sample_type + '_' + neuron_nums_str + '.csv', index=None)

    train_critics_df = pd.DataFrame(data=train_critics)
    train_critics_df.to_csv(submission_path + 'train_critics_' + str(args.ensemble_nums) + '_'
                            + args.sample_type + '_' + neuron_nums_str + '.csv', index=None)

    final_subs = np.mean(test_predict_arrs, axis=0)
    final_auc = roc_auc_score(test_data[:, 0: 1].tolist(), final_subs.tolist())

    rl_ensemble_preds_df = pd.DataFrame(data=final_subs)
    rl_ensemble_preds_df.to_csv(submission_path + 'submission_' + str(args.ensemble_nums) + '_'
                                + args.sample_type + '_' + neuron_nums_str + '.csv')

    rl_ensemble_aucs = [[final_auc]]
    rl_ensemble_aucs_df = pd.DataFrame(data=rl_ensemble_aucs)
    rl_ensemble_aucs_df.to_csv(submission_path + 'ensemble_aucs_' + str(args.ensemble_nums) + '_'
                               + args.sample_type + '_' + neuron_nums_str + '.csv', header=None)

    if args.dataset_name == 'ipinyou/':
        logger.info('Dataset {}, campain {}, models {}, ensemble auc {}\n'.format(args.dataset_name,
                                                                                  args.campaign_id,
                                                                                  args.rl_model_name, final_auc))
    else:
        logger.info(
            'Dataset {}, models {}, ensemble auc {}\n'.format(args.dataset_name, args.rl_model_name, final_auc))