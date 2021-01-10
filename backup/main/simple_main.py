import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
from sklearn.metrics import roc_auc_score
import src.models.DDPG_model as Model
import src.models.creat_data as Data
import src.models.p_model as p_Model

import torch
import torch.nn as nn
import torch.utils.data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(action_nums, feature_nums, field_nums, latent_dims, batch_size, memory_size, device, campaign_id):
    return Model.DDPG(feature_nums, field_nums, action_nums, latent_dims,
                      campaign_id=campaign_id, batch_size=batch_size, memory_size=memory_size, device=device)

def get_dataset(datapath, dataset_name, campaign_id, valid_day, test_day):
    data_path = datapath + dataset_name + campaign_id
    data_file_name = 'train.txt'
    day_index_file_name = 'day_index.csv'

    train_fm = pd.read_csv(data_path + data_file_name, header=None).values.astype(int)

    field_nums = len(train_fm[0, 1:])  # 特征域的数量
    feature_nums = np.max(train_fm[:, 1:].flatten()) + 1  # 特征数量

    day_indexs = pd.read_csv(data_path + day_index_file_name, header=None).values
    days = day_indexs[:, 0]  # 数据集中有的日期
    days_list = days.tolist()
    days_list.pop(days_list.index(valid_day))
    days_list.pop(days_list.index(test_day))

    train_data = np.array([])
    for i, day in enumerate(days_list):  # 生成训练集
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

def reward_functions(y_preds, features, FFM, labels, device):
    FFM_preds = FFM(features.cpu()).to(device).detach()

    reward = 1
    punishment = -1

    with_clk_indexs = (labels == 1).nonzero()[:, 0]
    without_clk_indexs = (labels == 0).nonzero()[:, 0]

    tensor_for_noclk = torch.ones(size=[len(without_clk_indexs), 1]).to(device)
    tensor_for_clk = torch.ones(size=[len(with_clk_indexs), 1]).to(device)

    # deviation_without_clk = FFM_preds[without_clk_indexs] - y_preds[without_clk_indexs]
    # reward_for_without_clk = torch.where(deviation_without_clk != 0, reward / deviation_without_clk, tensor_for_noclk * reward)
    # punishment_for_without_clk = torch.where(deviation_without_clk != 0, deviation_without_clk, tensor_for_noclk * punishment)
    #
    # deviation_with_clk = FFM_preds[with_clk_indexs] - y_preds[with_clk_indexs]
    # reward_for_with_clk = torch.where(deviation_with_clk != 0, -deviation_with_clk, tensor_for_clk * reward)
    # punishment_for_with_clk = torch.where(deviation_with_clk != 0, -deviation_with_clk, tensor_for_clk * punishment)

    reward_without_clk = torch.where(y_preds[without_clk_indexs] >= FFM_preds[without_clk_indexs], tensor_for_noclk * punishment, tensor_for_noclk * reward).cpu().numpy()
    reward_with_clk = torch.where(y_preds[with_clk_indexs] >= FFM_preds[with_clk_indexs], tensor_for_clk * reward, tensor_for_clk * punishment).cpu().numpy()

    for i, clk_index in enumerate(with_clk_indexs.cpu().numpy()):
        reward_without_clk = np.insert(reward_without_clk, clk_index, reward_with_clk[i]) # 向指定位置插入具有点击的奖励值

    return_reward = torch.FloatTensor(reward_without_clk).view(-1, 1)

    return return_reward

def train(model, FFM, data_loader, device, ou_noise_obj, exploration_rate):
    total_loss = 0
    log_intervals = 0
    total_rewards = 0
    for i, (features, labels) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        features, labels = features.long().to(device), torch.unsqueeze(labels, 1).float().to(device)
        ou_noise = ou_noise_obj()[:len(features)].reshape(-1, 1)

        actions = model.choose_action(features) # ctrs

        y_preds = torch.FloatTensor(np.clip(np.random.normal(actions, exploration_rate), 1e-5, 1)).to(device)
        rewards = reward_functions(y_preds, features, FFM, labels, device).to(device)

        action_rewards = torch.cat([y_preds, rewards], dim=1)

        model.store_transition(features, action_rewards)

        b_s, b_a, b_r, b_s_ = model.sample_batch()

        td_error = model.learn_c(b_s, b_a, b_r, b_s_)
        a_loss = model.learn_a(b_s)
        model.soft_update(model.Actor, model.Actor_)
        model.soft_update(model.Critic, model.Critic_)

        total_loss += td_error
        log_intervals += 1

        total_rewards += torch.sum(rewards, dim=0)

    return total_loss / log_intervals, total_rewards.cpu().numpy()[0] / log_intervals

def test(model, data_loader, loss, device):
    targets, predicts = list(), list()
    intervals = 0
    total_test_loss = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            y = torch.FloatTensor(model.choose_action(features)).to(device)

            test_loss = loss(y, labels.float())
            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
            intervals += 1
            total_test_loss += test_loss.item()

    return roc_auc_score(targets, predicts), total_test_loss / intervals


def submission(model, data_loader, device):
    targets, predicts = list(), list()
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            y = model.choose_action(features)

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())

    return predicts, roc_auc_score(targets, predicts)


def main(data_path, dataset_name, campaign_id, valid_day, test_day, action_nums, latent_dims, model_name, epoch, early_stop_type, batch_size, device, save_param_dir, ou_noise):
    if not os.path.exists(save_param_dir + campaign_id):
        os.mkdir(save_param_dir + campaign_id)

    device = torch.device(device)  # 指定运行设备
    train_fm, day_indexs, train_data, valid_data, test_data, field_nums, feature_nums = get_dataset(data_path,
                                                                                                    dataset_name,
                                                                                                    campaign_id,
                                                                                                    valid_day, test_day)

    train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    valid_dataset = Data.libsvm_dataset(valid_data[:, 1:], valid_data[:, 0])
    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    memory_size = round(len(train_data), -6)
    model = get_model(action_nums, feature_nums, field_nums, latent_dims, batch_size, memory_size, device, campaign_id)
    loss = nn.BCELoss()

    FFM = p_Model.FFM(feature_nums, field_nums, latent_dims)
    FFM.load_state_dict(model.embedding_layer.pretrain_params)

    valid_aucs = []
    valid_losses = []
    early_stop_index = 0
    is_early_stop = False

    start_time = datetime.datetime.now()
    exploration_rate = 1
    for epoch_i in range(epoch):
        torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

        train_start_time = datetime.datetime.now()

        train_average_loss, train_rewards = train(model, FFM, train_data_loader, device, ou_noise, exploration_rate)

        torch.save(model.Actor.state_dict(), save_param_dir + campaign_id + model_name + str(np.mod(epoch_i, 5)) + '.pth')

        auc, valid_loss = test(model, valid_data_loader, loss, device)
        valid_aucs.append(auc)
        valid_losses.append(valid_loss)

        train_end_time = datetime.datetime.now()
        print('epoch:', epoch_i, 'training average loss:', train_average_loss, 'training rewards', train_rewards, 'validation auc:', auc,
              'validation loss:', valid_loss, '[{}s]'.format((train_end_time - train_start_time).seconds))

        exploration_rate *= 0.95

        if eva_stopping(valid_aucs, valid_losses, early_stop_type):
            early_stop_index = np.mod(epoch_i - 4, 5)
            is_early_stop = True
            break

    end_time = datetime.datetime.now()

    if is_early_stop:
        test_model = get_model(action_nums, feature_nums, field_nums, latent_dims, batch_size, memory_size, device, campaign_id)
        load_path = save_param_dir + campaign_id + model_name + str(early_stop_index) + '.pth'

        test_model.Actor.load_state_dict(torch.load(load_path, map_location=device))  # 加载最优参数
    else:
        test_model = model

    torch.save(model.Actor.state_dict(), save_param_dir + model_name + 'best.pth')  # 存储最优参数

    submission_path = data_path + dataset_name + campaign_id + model_name + '/'  # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    # 验证集submission
    valid_predicts, valid_auc = submission(test_model, valid_data_loader, device)
    valid_pred_df = pd.DataFrame(data=valid_predicts)

    valid_pred_df.to_csv(submission_path + str(valid_day) + '_test_submission.csv', header=None)

    # 测试集submission
    test_predicts, test_auc = submission(test_model, test_data_loader, device)
    test_pred_df = pd.DataFrame(data=test_predicts)

    test_pred_df.to_csv(submission_path + str(test_day) + '_test_submission.csv', header=None)

    day_aucs = [[valid_day, valid_auc], [test_day, test_auc]]
    day_aucs_df = pd.DataFrame(data=day_aucs)
    day_aucs_df.to_csv(submission_path + 'day_aucs.csv', header=None)

    print('\ntest auc:', test_auc, datetime.datetime.now(), '[{}s]'.format((end_time - start_time).seconds))


def eva_stopping(valid_aucs, valid_losses, type):  # early stopping
    if type == 'auc':
        if len(valid_aucs) >= 5:
            if valid_aucs[-1] <= valid_aucs[-2] and valid_aucs[-2] <= valid_aucs[-3] and valid_aucs[-3] <= valid_aucs[
                -4] and valid_aucs[-4] <= valid_aucs[-5]:
                return True
    else:
        if len(valid_losses) >= 5:
            if valid_losses[-1] >= valid_losses[-2] and valid_losses[-2] >= valid_losses[-3] and valid_losses[-3] >= \
                    valid_losses[-4] and valid_losses[-4] >= valid_losses[-5]:
                return True

    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi')
    parser.add_argument('--valid_day', default=11, help='6, 7, 8, 9, 10, 11, 12')
    parser.add_argument('--test_day', default=12, help='6, 7, 8, 9, 10, 11, 12')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 3358, 3386, 3427, 3476')
    parser.add_argument('--model_name', default='DDPG', help='LR, FM, FFM')
    parser.add_argument('--action_nums', default=1)
    parser.add_argument('--latent_dims', default=5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop_type', default='loss', help='auc, loss')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_param_dir', default='../models/model_params/')

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(1)

    ou_noise = Model.OrnsteinUhlenbeckNoise(mu=np.zeros(args.batch_size))

    main(
        args.data_path,
        args.dataset_name,
        args.campaign_id,
        args.valid_day,
        args.test_day,
        args.action_nums,
        args.latent_dims,
        args.model_name,
        args.epoch,
        args.early_stop_type,
        args.batch_size,
        args.device,
        args.save_param_dir,
        ou_noise
    )