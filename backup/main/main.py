import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
from sklearn.metrics import roc_auc_score
# import src.models.PG_model as Model
import src.models.DDQN_model as Model
import src.models.p_model as p_model
import src.models.DDPG_for_PG_model as DDPG_for_PG_model
import src.models.creat_data as Data

import torch
import torch.nn as nn
import torch.utils.data
np.seterr(all='raise')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(action_nums, feature_nums, field_nums, latent_dims, batch_size, memory_size, device, campaign_id):

    ddqn_model = Model.DoubleDQN(feature_nums, field_nums, latent_dims,
                                    campaign_id=campaign_id, action_nums=action_nums, memory_size=memory_size, batch_size=batch_size, device=device)
    ddpg_for_pg_Model = DDPG_for_PG_model.DDPG(feature_nums, field_nums, latent_dims,
                                               action_nums=action_nums,
                                               campaign_id=campaign_id, batch_size=batch_size,
                                               memory_size=memory_size, device=device)
    return ddqn_model, ddpg_for_pg_Model


def get_dataset(datapath, dataset_name, campaign_id, valid_day, test_day):
    data_path = datapath + dataset_name + campaign_id
    data_file_name = 'train.txt'
    day_index_file_name = 'day_index.csv'

    train_fm = pd.read_csv(data_path + data_file_name, header=None).values.astype(int)

    field_nums = len(train_fm[0, 1:]) # 特征域的数量
    feature_nums = np.max(train_fm[:, 1:].flatten()) + 1 # 特征数量

    day_indexs = pd.read_csv(data_path + day_index_file_name, header=None).values
    days = day_indexs[:, 0] # 数据集中有的日期
    days_list = days.tolist()
    days_list.pop(days_list.index(valid_day))
    days_list.pop(days_list.index(test_day))

    train_data = np.array([])
    for i, day in enumerate(days_list): # 生成训练集
        current_day_index = day_indexs[days == day]
        data_index_start = current_day_index[0, 1]
        data_index_end = current_day_index[0, 2] + 1

        data_ = train_fm[data_index_start: data_index_end, :]
        if i == 0:
            train_data = data_
        else:
            train_data = np.concatenate((train_data, data_), axis=0)

    # 生成验证集
    valid_day_index = day_indexs[days == valid_day]
    valid_index_start = valid_day_index[0, 1]
    valid_index_end = valid_day_index[0, 2] + 1

    valid_data = train_fm[valid_index_start: valid_index_end, :]

    # 生成测试集
    test_day_index = day_indexs[days == test_day]
    test_index_start = test_day_index[0, 1]
    test_index_end = test_day_index[0, 2] + 1
    test_data = train_fm[test_index_start: test_index_end, :]

    return train_fm, day_indexs, train_data, valid_data, test_data, field_nums, feature_nums


# def generate_preds(model_dict, features, actions, prob_weights, labels, device, mode):
#     y_preds = torch.ones(size=[len(features), 1]).to(device)
#     rewards = torch.ones(size=[len(features), 1]).to(device)
#
#     origin_prob_weights = prob_weights
#     if mode == 'train':
#         prob_weights = torch.softmax(prob_weights, dim=1)
#
#     sort_prob_weights, sortindex_prob_weights = torch.sort(-prob_weights, dim=1)
#
#     pretrain_model_len = len(model_dict) # 有多少个预训练模型
#
#     pretrain_y_preds = {}
#     for i in range(pretrain_model_len):
#         pretrain_y_preds[i] = model_dict[i](features).detach()
#
#     for i in range(pretrain_model_len): # 根据ddqn_model的action,判断要选择ensemble的数量
#         with_action_indexs = (actions == (i + 1)).nonzero()[:, 0]
#         current_choose_models = sortindex_prob_weights[with_action_indexs][:, :i + 1]
#         current_basic_rewards = torch.ones(size=[len(with_action_indexs), 1]).to(device)
#
#         current_with_clk_indexs = (labels[with_action_indexs] == 1).nonzero()[:, 0]
#         current_without_clk_indexs = (labels[with_action_indexs] == 0).nonzero()[:, 0]
#
#         if i == 0:
#             current_y_preds = torch.ones(size=[len(with_action_indexs), 1]).to(device)
#             # current_origin_prob_weights, current_origin_sortindex_prob_weights = torch.sort(
#             #     origin_prob_weights[with_action_indexs], dim=1)
#             # current_origin_prob_weights = current_origin_prob_weights.to(device)
#             for k in range(pretrain_model_len):
#                 current_pretrain_y_preds = pretrain_y_preds[k][with_action_indexs]
#                 choose_model_indexs = (current_choose_models == k).nonzero()[:, 0]
#                 current_y_preds[choose_model_indexs, :] = current_pretrain_y_preds[choose_model_indexs]
#                 # current_y_preds[choose_model_indexs, :] = torch.mul(
#                 #     current_origin_prob_weights[choose_model_indexs][:, pretrain_model_len - 1].view(-1, 1),
#                 #     current_pretrain_y_preds[choose_model_indexs])
#
#             y_preds[with_action_indexs, :] = current_y_preds
#
#             with_clk_rewards = torch.where(
#                 current_y_preds[current_with_clk_indexs] >= pretrain_y_preds[pretrain_model_len - 1][with_action_indexs][
#                     current_with_clk_indexs],
#                 current_basic_rewards[current_with_clk_indexs] * 1,
#                 current_basic_rewards[current_with_clk_indexs] * -1
#             )
#
#             without_clk_rewards = torch.where(
#                 current_y_preds[current_without_clk_indexs] <= pretrain_y_preds[pretrain_model_len - 1][with_action_indexs][
#                     current_without_clk_indexs],
#                 current_basic_rewards[current_without_clk_indexs] * 1,
#                 current_basic_rewards[current_without_clk_indexs] * -1
#             )
#
#             current_basic_rewards[current_with_clk_indexs] = with_clk_rewards
#             current_basic_rewards[current_without_clk_indexs] = without_clk_rewards
#
#             rewards[with_action_indexs, :] = current_basic_rewards
#         elif i == pretrain_model_len - 1:
#             current_prob_weights = prob_weights[with_action_indexs].to(device)
#             current_pretrain_y_preds = torch.cat([
#                 pretrain_y_preds[l][with_action_indexs] for l in range(pretrain_model_len)
#             ], dim=1)
#             current_y_preds = torch.sum(torch.mul(current_prob_weights, current_pretrain_y_preds), dim=1).view(-1, 1)
#
#             y_preds[with_action_indexs, :] = current_y_preds
#
#             with_clk_rewards = torch.where(
#                 current_y_preds[current_with_clk_indexs] >= current_pretrain_y_preds[
#                     current_with_clk_indexs].mean(dim=1).view(-1, 1),
#                 current_basic_rewards[current_with_clk_indexs] * 1,
#                 current_basic_rewards[current_with_clk_indexs] * -1
#             )
#
#             without_clk_rewards = torch.where(
#                 current_y_preds[current_without_clk_indexs] <= current_pretrain_y_preds[
#                     current_without_clk_indexs].mean(dim=1).view(-1, 1),
#                 current_basic_rewards[current_without_clk_indexs] * 1,
#                 current_basic_rewards[current_without_clk_indexs] * -1
#             )
#
#             current_basic_rewards[current_with_clk_indexs] = with_clk_rewards
#             current_basic_rewards[current_without_clk_indexs] = without_clk_rewards
#
#             rewards[with_action_indexs, :] = current_basic_rewards
#         else:
#             current_softmax_weights = torch.softmax(
#                 sort_prob_weights[with_action_indexs][:, :i + 1], dim=1
#             ).to(device)  # 再进行softmax
#
#             current_row_preds = torch.ones(size=[len(with_action_indexs), i + 1]).to(device)
#             for m in range(i+1):
#                 current_row_choose_models = current_choose_models[:, m:m+1]
#                 for k in range(pretrain_model_len):
#                     current_pretrain_y_preds = pretrain_y_preds[k][with_action_indexs]
#                     choose_model_indexs = (current_row_choose_models == k).nonzero()[:, 0]
#
#                     current_row_preds[choose_model_indexs, m:m+1] = current_pretrain_y_preds[choose_model_indexs]
#
#             current_y_preds = torch.sum(torch.mul(current_softmax_weights, current_row_preds), dim=1).view(-1, 1)
#             y_preds[with_action_indexs, :] = current_y_preds
#
#             with_clk_rewards = torch.where(
#                 current_y_preds[current_with_clk_indexs] >= current_row_preds[
#                     current_with_clk_indexs].mean(dim=1).view(-1, 1),
#                 current_basic_rewards[current_with_clk_indexs] * 1,
#                 current_basic_rewards[current_with_clk_indexs] * -1
#             )
#
#             without_clk_rewards = torch.where(
#                 current_y_preds[current_without_clk_indexs] <= current_row_preds[
#                     current_without_clk_indexs].mean(dim=1).view(-1, 1),
#                 current_basic_rewards[current_without_clk_indexs] * 1,
#                 current_basic_rewards[current_without_clk_indexs] * -1
#             )
#
#             current_basic_rewards[current_with_clk_indexs] = with_clk_rewards
#             current_basic_rewards[current_without_clk_indexs] = without_clk_rewards
#
#             rewards[with_action_indexs, :] = current_basic_rewards
#
#     return y_preds, prob_weights.to(device), rewards

def generate_preds(model_dict, features, actions, prob_weights, labels, device, mode):
    y_preds = torch.ones(size=[len(features), 1]).to(device)
    rewards = torch.ones(size=[len(features), 1]).to(device)

    if mode == 'train':
        prob_weights = torch.softmax(prob_weights, dim=1)

    sort_prob_weights, sortindex_prob_weights = torch.sort(-prob_weights, dim=1)

    pretrain_model_len = len(model_dict) # 有多少个预训练模型

    return_prob_weights = torch.zeros(size=[len(features), pretrain_model_len]).to(device)

    pretrain_y_preds = {}
    for i in range(pretrain_model_len):
        pretrain_y_preds[i] = model_dict[i](features).detach()

    choose_model_lens = range(2, pretrain_model_len + 1)
    for i in choose_model_lens: # 根据ddqn_model的action,判断要选择ensemble的数量
        with_action_indexs = (actions == i).nonzero()[:, 0]
        current_choose_models = sortindex_prob_weights[with_action_indexs][:, :i]
        current_basic_rewards = torch.ones(size=[len(with_action_indexs), 1]).to(device)
        current_prob_weights = prob_weights[with_action_indexs]

        current_with_clk_indexs = (labels[with_action_indexs] == 1).nonzero()[:, 0]
        current_without_clk_indexs = (labels[with_action_indexs] == 0).nonzero()[:, 0]

        if i == pretrain_model_len:
            current_pretrain_y_preds = torch.cat([
                pretrain_y_preds[l][with_action_indexs] for l in range(pretrain_model_len)
            ], dim=1)
            current_y_preds = torch.sum(torch.mul(current_prob_weights, current_pretrain_y_preds), dim=1).view(-1, 1)

            y_preds[with_action_indexs, :] = current_y_preds

            return_prob_weights[with_action_indexs] = current_prob_weights
        else:
            current_softmax_weights = torch.softmax(
                sort_prob_weights[with_action_indexs][:, :i] * -1, dim=1
            ).to(device)  # 再进行softmax

            for k in range(i):
                return_prob_weights[with_action_indexs, current_choose_models[:, k]] = current_softmax_weights[:, k]

            current_row_preds = torch.ones(size=[len(with_action_indexs), i]).to(device)
            for m in range(i):
                current_row_choose_models = current_choose_models[:, m:m+1]
                for k in range(pretrain_model_len):
                    current_pretrain_y_preds = pretrain_y_preds[k][with_action_indexs]
                    choose_model_indexs = (current_row_choose_models == k).nonzero()[:, 0]

                    current_row_preds[choose_model_indexs, m:m+1] = current_pretrain_y_preds[choose_model_indexs]

            current_y_preds = torch.sum(torch.mul(current_softmax_weights, current_row_preds), dim=1).view(-1, 1)
            y_preds[with_action_indexs, :] = current_y_preds

        # with_clk_rewards = torch.where(
        #     current_y_preds[current_with_clk_indexs] >= current_pretrain_y_preds[
        #         current_with_clk_indexs].mean(dim=1).view(-1, 1),
        #     current_y_preds[current_with_clk_indexs] - current_pretrain_y_preds[
        #         current_with_clk_indexs].mean(dim=1).view(-1, 1),
        #     current_pretrain_y_preds[
        #         current_with_clk_indexs].mean(dim=1).view(-1, 1) - current_y_preds[current_with_clk_indexs]
        # )
        #
        # without_clk_rewards = torch.where(
        #     current_y_preds[current_without_clk_indexs] <= current_pretrain_y_preds[
        #         current_without_clk_indexs].mean(dim=1).view(-1, 1),
        #     current_pretrain_y_preds[
        #         current_without_clk_indexs].mean(dim=1).view(-1, 1) - current_y_preds[current_without_clk_indexs],
        #     current_y_preds[current_without_clk_indexs] - current_pretrain_y_preds[
        #         current_without_clk_indexs].mean(dim=1).view(-1, 1)
        # )

        with_clk_rewards = torch.where(
            current_y_preds[current_with_clk_indexs] >= current_pretrain_y_preds[
                current_with_clk_indexs].mean(dim=1).view(-1, 1),
            current_basic_rewards[current_with_clk_indexs] * 1,
            current_basic_rewards[current_with_clk_indexs] * -1
        )

        without_clk_rewards = torch.where(
            current_y_preds[current_without_clk_indexs] <= current_pretrain_y_preds[
                current_without_clk_indexs].mean(dim=1).view(-1, 1),
            current_basic_rewards[current_without_clk_indexs] * 1,
            current_basic_rewards[current_without_clk_indexs] * -1
        )

        current_basic_rewards[current_with_clk_indexs] = with_clk_rewards
        current_basic_rewards[current_without_clk_indexs] = without_clk_rewards

        rewards[with_action_indexs, :] = current_basic_rewards

    return y_preds, return_prob_weights, rewards


def train(ddqn_model, ddpg_for_pg_model, model_dict, data_loader, ou_noise_obj, exploration_rate, device):
    total_loss = 0
    log_intervals = 0
    total_rewards = 0
    for i, (features, labels) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
        actions = ddqn_model.choose_action(features, exploration_rate)

        prob_weights = ddpg_for_pg_model.choose_action(features, actions.float(), exploration_rate)

        y_preds, prob_weights_new, rewards = \
            generate_preds(model_dict, features, actions, prob_weights, labels, device, mode='train')

        ddqn_model.store_transition(torch.cat([features, actions, rewards.long()], dim=1))
        action_rewards = torch.cat([prob_weights_new, rewards], dim=1)
        ddpg_for_pg_model.store_transition(features, action_rewards, actions.float())

        ddqn_model.learn()

        b_s, b_a, b_r, b_s_, b_pg_a = ddpg_for_pg_model.sample_batch()
        td_error = ddpg_for_pg_model.learn_c(b_s, b_a, b_r, b_s_, b_pg_a)
        a_loss = ddpg_for_pg_model.learn_a(b_s, b_pg_a)
        ddpg_for_pg_model.soft_update(ddpg_for_pg_model.Actor, ddpg_for_pg_model.Actor_)
        ddpg_for_pg_model.soft_update(ddpg_for_pg_model.Critic, ddpg_for_pg_model.Critic_)

        total_loss += td_error # 取张量tensor里的标量值，如果直接返回train_loss很可能会造成GPU out of memory
        log_intervals += 1

        total_rewards += torch.sum(rewards, dim=0).item()

        torch.cuda.empty_cache()# 清除缓存

    return total_loss / log_intervals, total_rewards / log_intervals

def test(ddqn_model, ddpg_for_pg_model, model_dict, data_loader, loss, device):
    targets, predicts = list(), list()
    intervals = 0
    total_test_loss = 0
    with torch.no_grad():
        for i, (features, labels) in enumerate(data_loader):
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            actions = ddqn_model.choose_best_action(features)
            prob_weights = ddpg_for_pg_model.choose_best_action(features, actions.float())

            # x = torch.argsort(prob_weights)[:, 0]
            # print(len((actions == 2).nonzero()), len((x == 3).nonzero()), len((x == 4).nonzero()), len((x == 5).nonzero()))
            #
            # print(len((x == 0).nonzero()), len((x == 1).nonzero()), len((x == 2).nonzero()), len((x == 3).nonzero()), len((x == 4).nonzero()))
            y, prob_weights_new, rewards = generate_preds(model_dict, features, actions, prob_weights, labels, device, mode='test')

            test_loss = loss(y, labels.float())
            targets.extend(labels.tolist()) # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
            intervals += 1
            total_test_loss += test_loss.item()

    return roc_auc_score(targets, predicts), total_test_loss / intervals


def submission(ddqn_model, ddpg_for_pg_model, model_dict, data_loader, device):
    targets, predicts = list(), list()
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            actions = ddqn_model.choose_best_action(features)
            prob_weights = ddpg_for_pg_model.choose_best_action(features, actions.float())
            y, prob_weights_new, rewards = generate_preds(model_dict, features, actions, prob_weights, labels, device, mode='test')

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())

    return predicts, roc_auc_score(targets, predicts)


def main(data_path, dataset_name, campaign_id, valid_day, test_day, latent_dims, model_name, epoch, learning_rate,
         weight_decay, early_stop_type, batch_size, device, save_param_dir, ou_noise_obj):
    if not os.path.exists(save_param_dir):
        os.mkdir(save_param_dir)

    device = torch.device(device) # 指定运行设备
    train_fm, day_indexs, train_data, valid_data, test_data, field_nums, feature_nums = get_dataset(data_path, dataset_name, campaign_id, valid_day, test_day)

    train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    valid_dataset = Data.libsvm_dataset(valid_data[:, 1:], valid_data[:, 0])
    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    # FFM = p_model.FFM(feature_nums, field_nums, latent_dims)
    # FFM_pretain_params = torch.load('models/model_params/' + campaign_id + 'FFMbest.pth')
    # FFM.load_state_dict(FFM_pretain_params)
    # FFM.eval()

    FM = p_model.FM(feature_nums, latent_dims)
    FM_pretain_params = torch.load('models/model_params/' + campaign_id + 'FMbest.pth')
    FM.load_state_dict(FM_pretain_params)
    FM.eval()

    WandD = p_model.WideAndDeep(feature_nums, field_nums, latent_dims)
    WandD_pretrain_params = torch.load('models/model_params/' + campaign_id + 'W&Dbest.pth')
    WandD.load_state_dict(WandD_pretrain_params)
    WandD.eval()

    DeepFM = p_model.DeepFM(feature_nums, field_nums, latent_dims)
    DeepFM_pretrain_params = torch.load('models/model_params/' + campaign_id + 'DeepFMbest.pth')
    DeepFM.load_state_dict(DeepFM_pretrain_params)
    DeepFM.eval()

    FNN = p_model.FNN(feature_nums, field_nums, latent_dims)
    FNN_pretrain_params = torch.load('models/model_params/' + campaign_id + 'FNNbest.pth')
    FNN.load_embedding(FM_pretain_params)
    FNN.load_state_dict(FNN_pretrain_params)
    FNN.eval()

    IPNN = p_model.InnerPNN(feature_nums, field_nums, latent_dims)
    IPNN_pretrain_params = torch.load('models/model_params/' + campaign_id + 'IPNNbest.pth')
    IPNN.load_embedding(FM_pretain_params)
    IPNN.load_state_dict(IPNN_pretrain_params)
    IPNN.eval()

    DCN = p_model.DCN(feature_nums, field_nums, latent_dims)
    DCN_pretrain_params = torch.load('models/model_params/' + campaign_id + 'DCNbest.pth')
    # DCN.load_embedding(FM_pretain_params)
    DCN.load_state_dict(DCN_pretrain_params)
    DCN.eval()

    model_dict = {0: IPNN.to(device), 1: WandD.to(device), 2: DeepFM.to(device), 3: FNN.to(device), 4: DCN.to(device)}
    # model_dict = {0: DeepFM.to(device), 1: WandD.to(device), 2: FFM.to(device),
    #               3: FNN.to(device)}

    # model_dict = {0: OPNN.to(device), 1: DeepFM.to(device), 2: FNN.to(device), 3: WandD.to(device), 4: FFM.to(device)}
    #
    model_dict_len = len(model_dict)

    memory_size = round(len(train_data), -6)
    ddqn_model, ddpg_for_pg_model = get_model(model_dict_len, feature_nums, field_nums, latent_dims, batch_size, memory_size, device, campaign_id)

    loss = nn.BCELoss()

    valid_aucs = []
    valid_losses = []
    early_stop_index = 0
    is_early_stop = False

    start_time = datetime.datetime.now()
    exploration_rate = 1
    for epoch_i in range(epoch):
        torch.cuda.empty_cache() # 清理无用的cuda中间变量缓存

        train_start_time = datetime.datetime.now()

        train_average_loss, train_average_rewards = train(ddqn_model, ddpg_for_pg_model, model_dict, train_data_loader, ou_noise_obj, exploration_rate, device)

        # torch.save(ddqn_model.eval_net.state_dict(), save_param_dir + 'ddqn_model' + str(np.mod(epoch_i, 5)) + '.pth')
        # torch.save(ddpg_for_pg_model.Actor.state_dict(), save_param_dir + 'ddpg_for_pg_model' + str(np.mod(epoch_i, 5)) + '.pth')

        auc, valid_loss = test(ddqn_model, ddpg_for_pg_model, model_dict, valid_data_loader, loss, device)
        test_auc_temp, test_loss = test(ddqn_model, ddpg_for_pg_model, model_dict, test_data_loader, loss, device)
        valid_aucs.append(auc)
        valid_losses.append(valid_loss)

        train_end_time = datetime.datetime.now()
        print('epoch:', epoch_i, 'training average loss:', train_average_loss, 'training average rewards',
              train_average_rewards, 'validation auc:', auc,
               'validation loss:', valid_loss, 'test auc', test_auc_temp, '[{}s]'.format((train_end_time - train_start_time).seconds))

        exploration_rate *= 0.95
        # if eva_stopping(valid_aucs, valid_losses, early_stop_type):
        #     early_stop_index = np.mod(epoch_i - 4, 5)
        #     is_early_stop = True
        #     break

    end_time = datetime.datetime.now()

    if is_early_stop:
        test_ddqn_model, test_ddpg_for_pg_model = get_model(model_dict_len, feature_nums, field_nums, latent_dims, batch_size, memory_size, device, campaign_id)
        load_pg_path = save_param_dir + 'ddqn_model' + str(early_stop_index) + '.pth'
        load_ddpg_path = save_param_dir + 'ddpg_for_pg_model' + str(early_stop_index) + '.pth'

        test_ddqn_model.eval_net.load_state_dict(torch.load(load_pg_path, map_location=device))  # 加载最优参数
        test_ddpg_for_pg_model.Actor.load_state_dict(torch.load(load_ddpg_path, map_location=device))  # 加载最优参数
    else:
        test_ddqn_model = ddqn_model
        test_ddpg_for_pg_model = ddpg_for_pg_model

    auc, test_loss = test(test_ddqn_model, test_ddpg_for_pg_model, model_dict, test_data_loader, loss, device)
    print('\ntest auc:', auc, datetime.datetime.now(), '[{}s]'.format((end_time - start_time).seconds))

    submission_path = data_path + dataset_name + campaign_id + model_name + '/' # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    # 验证集submission
    valid_predicts, valid_auc = submission(test_ddqn_model, test_ddpg_for_pg_model, model_dict, valid_data_loader, device)
    valid_pred_df = pd.DataFrame(data=valid_predicts)

    valid_pred_df.to_csv(submission_path + str(valid_day) + '_test_submission.csv', header=None)

    # 测试集submission
    test_predicts, test_auc = submission(test_ddqn_model, test_ddpg_for_pg_model, model_dict, test_data_loader, device)
    test_pred_df = pd.DataFrame(data=test_predicts)

    test_pred_df.to_csv(submission_path + str(test_day) + '_test_submission.csv', header=None)

    day_aucs = [[valid_day, valid_auc], [test_day, test_auc]]
    day_aucs_df = pd.DataFrame(data=day_aucs)
    day_aucs_df.to_csv(submission_path + 'day_aucs.csv', header=None)

    torch.save(test_ddqn_model.eval_net.state_dict(), save_param_dir + campaign_id + '/ddqn_model' + 'best.pth')  # 存储最优参数
    torch.save(test_ddpg_for_pg_model.Actor.state_dict(), save_param_dir + campaign_id + '/ddpg_for_pg_model' + 'best.pth')  # 存储最优参数


def eva_stopping(valid_aucs, valid_losses, type): # early stopping
    if type == 'auc':
        if len(valid_aucs) > 5:
            if valid_aucs[-1] < valid_aucs[-2] and valid_aucs[-2] < valid_aucs[-3] and valid_aucs[-3] < valid_aucs[-4] and valid_aucs[-4] < valid_aucs[-5]:
                return True
    else:
        if len(valid_losses) > 5:
            if valid_losses[-1] > valid_losses[-2] and valid_losses[-2] > valid_losses[-3] and valid_losses[-3] > valid_losses[-4] and valid_losses[-4] > valid_losses[-5]:
                return True

    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi')
    parser.add_argument('--valid_day', default=11, help='6, 7, 8, 9, 10, 11, 12')
    parser.add_argument('--test_day', default=12, help='6, 7, 8, 9, 10, 11, 12')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 3386')
    parser.add_argument('--model_name', default='PG_DDPG', help='LR, FM, FFM, W&D')
    parser.add_argument('--latent_dims', default=8)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop_type', default='auc', help='auc, loss')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_param_dir', default='../models/model_params/')

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(1)

    ou_noise = DDPG_for_PG_model.OrnsteinUhlenbeckNoise(mu=np.zeros(args.batch_size))

    main(
        args.data_path,
        args.dataset_name,
        args.campaign_id,
        args.valid_day,
        args.test_day,
        args.latent_dims,
        args.model_name,
        args.epoch,
        args.learning_rate,
        args.weight_decay,
        args.early_stop_type,
        args.batch_size,
        args.device,
        args.save_param_dir,
        ou_noise
    )