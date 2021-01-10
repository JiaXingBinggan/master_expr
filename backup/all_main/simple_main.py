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
import src.models.p_model as p_model

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
    return Model.DDPG(feature_nums, field_nums, latent_dims, action_nums,
                      campaign_id=campaign_id, batch_size=batch_size, memory_size=memory_size, device=device)


def get_dataset(datapath, dataset_name, campaign_id):
    data_path = datapath + dataset_name + campaign_id
    train_data_file_name = 'train_.txt'
    train_fm = pd.read_csv(data_path + train_data_file_name, header=None).values.astype(int)

    test_data_file_name = 'test_.txt'
    test_fm = pd.read_csv(data_path + test_data_file_name, header=None).values.astype(int)

    field_nums = len(train_fm[0, 1:])  # 特征域的数量

    feature_index_name = 'featindex.txt'
    feature_index = pd.read_csv(data_path + feature_index_name, header=None).values
    feature_nums = int(feature_index[-1, 0].split('\t')[1]) + 1  # 特征数量

    train_data = train_fm
    test_data = test_fm

    return train_fm, train_data, test_data, field_nums, feature_nums


def reward_functions(y_preds, best_origin_model, features, labels, device):
    best_origin_model_preds = best_origin_model(features).detach()

    current_reward = torch.ones(size=[len(y_preds), 1]).to(device)

    # reward = 100
    # punishment = -100

    with_clk_indexs = (labels == 1).nonzero()[:, 0]
    without_clk_indexs = (labels == 0).nonzero()[:, 0]

    # tensor_for_noclk = torch.ones(size=[len(without_clk_indexs), 1]).to(device)
    # tensor_for_clk = torch.ones(size=[len(with_clk_indexs), 1]).to(device)

    reward_without_clk = 0.1 - y_preds[without_clk_indexs]
    # reward_without_clk = torch.where(y_preds[without_clk_indexs] >= best_origin_model_preds[without_clk_indexs],
    #                                  tensor_for_noclk * punishment,
    #                                  tensor_for_noclk * reward)
    reward_with_clk = y_preds[with_clk_indexs] - 0.9
    # reward_with_clk = torch.where(y_preds[with_clk_indexs] >= best_origin_model_preds[with_clk_indexs],
    #                               tensor_for_clk * reward,
    #                               tensor_for_clk * punishment)

    current_reward[with_clk_indexs] = reward_with_clk * 10
    current_reward[without_clk_indexs] = reward_without_clk * 10
    # print(current_reward[with_clk_indexs], y_preds[with_clk_indexs])
    return current_reward


def train(model, best_origin_model, data_loader, device, exploration_rate, ou_noise_obj):
    total_loss = 0
    log_intervals = 0
    total_rewards = 0

    targets = list()
    predicts = list()
    for i, (features, labels) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        features, labels = features.long().to(device), torch.unsqueeze(labels, 1).float().to(device)
        # ou_noise = torch.FloatTensor(ou_noise_obj()[: len(features)]).unsqueeze(1).to(device)

        y_preds, actions = model.choose_action(features, labels, exploration_rate) # ctrs

        rewards = reward_functions(y_preds, best_origin_model, features, labels, device)

        action_rewards = torch.cat([actions, rewards], dim=1)

        model.store_transition(features, action_rewards)

        b_s, b_a, b_r, b_s_ = model.sample_batch()

        td_error = model.learn_c(b_s, b_a, b_r, b_s_)
        a_loss = model.learn_a(b_s)
        model.soft_update(model.Actor, model.Actor_)
        model.soft_update(model.Critic, model.Critic_)

        total_loss += td_error
        log_intervals += 1

        total_rewards += torch.sum(rewards, dim=0)

        targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
        predicts.extend(y_preds.tolist())

    return total_loss / log_intervals, total_rewards.cpu().numpy()[0] / log_intervals, roc_auc_score(targets, predicts)

def test(model, data_loader, loss, device):
    targets, predicts = list(), list()
    intervals = 0
    total_test_loss = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            y = model.choose_best_action(features)
            # print(y)
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
            y = model.choose_best_action(features)

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())

    return predicts, roc_auc_score(targets, predicts)


def main(data_path, dataset_name, campaign_id, action_nums, latent_dims, model_name, epoch, early_stop_type, batch_size, device, save_param_dir, ou_noise_obj):
    if not os.path.exists(save_param_dir + campaign_id):
        os.mkdir(save_param_dir + campaign_id)
    device = torch.device(device)  # 指定运行设备
    train_fm, train_data, test_data, field_nums, feature_nums = get_dataset(data_path, dataset_name, campaign_id)

    train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    memory_size = round(len(train_data), -6)
    model = get_model(action_nums, feature_nums, field_nums, latent_dims, batch_size, memory_size, device, campaign_id)
    loss = nn.BCELoss()

    FM = p_model.FM(feature_nums, latent_dims)
    FM_pretrain_params = torch.load(save_param_dir + campaign_id + 'FMbest.pth')
    FM.load_state_dict(FM_pretrain_params)
    FM.eval()
    FM.to(device)

    valid_aucs = []
    valid_losses = []
    early_stop_index = 0
    is_early_stop = False

    start_time = datetime.datetime.now()
    exploration_rate = 1e-1
    for epoch_i in range(epoch):
        torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

        train_start_time = datetime.datetime.now()

        train_average_loss, train_rewards, train_auc = train(model, FM, train_data_loader, device, exploration_rate, ou_noise_obj)

        # torch.save(model.Actor.state_dict(), save_param_dir + campaign_id + model_name + str(np.mod(epoch_i, 5)) + '.pth')

        auc, valid_loss = test(model, test_data_loader, loss, device)
        valid_aucs.append(auc)
        valid_losses.append(valid_loss)

        train_end_time = datetime.datetime.now()
        print('epoch:', epoch_i, 'training average loss:', train_average_loss, 'training rewards', train_rewards,
              'training auc', train_auc, 'validation auc:', auc,
              'validation loss:', valid_loss, '[{}s]'.format((train_end_time - train_start_time).seconds))

        exploration_rate /= 1.01

        # if eva_stopping(valid_aucs, valid_losses, early_stop_type):
        #     early_stop_index = np.mod(epoch_i - 4, 5)
        #     is_early_stop = True
        #     break

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

    # 测试集submission
    test_predicts, test_auc = submission(test_model, test_data_loader, device)
    test_pred_df = pd.DataFrame(data=test_predicts)

    test_pred_df.to_csv(submission_path + 'test_submission.csv', header=None)

    day_aucs = [[test_auc]]
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
    parser.add_argument('--campaign_id', default='1458/', help='1458, 3358, 3386, 3427, 3476')
    parser.add_argument('--model_name', default='DDPG', help='LR, FM, FFM')
    parser.add_argument('--action_nums', default=1)
    parser.add_argument('--latent_dims', default=10)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop_type', default='loss', help='auc, loss')
    parser.add_argument('--batch_size', type=int, default=128)
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