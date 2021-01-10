import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
from sklearn.metrics import roc_auc_score
import src.models.p_model as Model
import src.models.v10_Hybrid_TD3_model_PER as td3_model
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


def get_model(action_nums, feature_nums, field_nums, latent_dims, init_lr_a, init_lr_c, data_len, train_batch_size, memory_size, random_seed, device, campaign_id):
    RL_model = td3_model.Hybrid_TD3_Model(feature_nums, field_nums, latent_dims,
                                          action_nums=action_nums, lr_C_A=init_lr_a, lr_D_A=init_lr_a, lr_C=init_lr_c,
                                          data_len=data_len, batch_size=train_batch_size,
                                          campaign_id=campaign_id,
                                          memory_size=memory_size, random_seed=random_seed, device=device)

    return RL_model

def generate_preds(model_dict, features, actions, prob_weights, c_actions,
                   labels, device, mode):
    y_preds = torch.ones_like(actions, dtype=torch.float)
    rewards = torch.ones_like(actions, dtype=torch.float)

    sort_prob_weights, sortindex_prob_weights = torch.sort(-prob_weights, dim=1)

    pretrain_model_len = len(model_dict)  # 有多少个预训练模型

    pretrain_y_preds = {}
    for i in range(pretrain_model_len):
        pretrain_y_preds[i] = model_dict[i](features).detach()

    return_c_actions = torch.zeros(size=(features.size()[0], len(model_dict))).to(device)

    choose_model_lens = range(1, pretrain_model_len + 1)
    for i in choose_model_lens:  # 根据ddqn_model的action,判断要选择ensemble的数量
        with_action_indexs = (actions == i).nonzero()[:, 0]
        current_choose_models = sortindex_prob_weights[with_action_indexs][:, :i]
        current_basic_rewards = torch.ones(size=[len(with_action_indexs), 1]).to(device) * 1
        current_prob_weights = prob_weights[with_action_indexs]

        current_with_clk_indexs = (labels[with_action_indexs] == 1).nonzero()[:, 0]
        current_without_clk_indexs = (labels[with_action_indexs] == 0).nonzero()[:, 0]

        current_pretrain_y_preds = torch.cat([
            pretrain_y_preds[l][with_action_indexs] for l in range(pretrain_model_len)
        ], dim=1)

        current_c_actions = c_actions[with_action_indexs, :]
        current_ctrs = current_pretrain_y_preds

        if i == pretrain_model_len:
            current_y_preds = torch.sum(torch.mul(current_prob_weights, current_pretrain_y_preds), dim=1).view(-1, 1)
            y_preds[with_action_indexs, :] = current_y_preds

            return_c_actions[with_action_indexs, :] = current_c_actions
        elif i == 1:
            current_y_preds = torch.ones(size=[len(with_action_indexs), 1]).to(device)
            current_c_actions_temp = torch.zeros(size=[len(with_action_indexs), len(model_dict)]).to(device)
            for k in range(pretrain_model_len):
                choose_model_indexs = (current_choose_models == k).nonzero()[:, 0] # 找出下标
                current_y_preds[choose_model_indexs, 0] = current_pretrain_y_preds[choose_model_indexs, k]

                current_c_actions_temp[choose_model_indexs, k] = current_c_actions[choose_model_indexs, k]
            y_preds[with_action_indexs, :] = current_y_preds
            return_c_actions[with_action_indexs, :] = current_c_actions_temp
        else:
            current_softmax_weights = torch.softmax(
                sort_prob_weights[with_action_indexs][:, :i] * -1, dim=1
            ).to(device)  # 再进行softmax

            current_row_preds = torch.ones(size=[len(with_action_indexs), i]).to(device)
            current_c_actions_temp = torch.zeros(size=[len(with_action_indexs), len(model_dict)]).to(device)

            for m in range(i):
                current_row_choose_models = current_choose_models[:, m:m + 1]  # 这个和current_c_actions等长

                for k in range(pretrain_model_len):
                    current_pretrain_y_pred = current_pretrain_y_preds[:, k: k + 1]
                    choose_model_indexs = (current_row_choose_models == k).nonzero()[:, 0]

                    current_row_preds[choose_model_indexs, m:m + 1] = current_pretrain_y_pred[choose_model_indexs]

                    current_c_actions_temp[choose_model_indexs, k:k + 1] = current_c_actions[choose_model_indexs,
                                                                           k: k + 1]

            current_y_preds = torch.sum(torch.mul(current_softmax_weights, current_row_preds), dim=1).view(-1, 1)
            y_preds[with_action_indexs, :] = current_y_preds

            return_c_actions[with_action_indexs, :] = current_c_actions_temp # 为了让没有使用到的位置,值置为0

        with_clk_rewards = torch.where(
            current_y_preds[current_with_clk_indexs] > current_pretrain_y_preds[
                current_with_clk_indexs].mean(dim=1).view(-1, 1),
            current_basic_rewards[current_with_clk_indexs] * 1,
            current_basic_rewards[current_with_clk_indexs] * 0
        )

        without_clk_rewards = torch.where(
            current_y_preds[current_without_clk_indexs] < current_pretrain_y_preds[
                current_without_clk_indexs].mean(dim=1).view(-1, 1),
            current_basic_rewards[current_without_clk_indexs] * 1,
            current_basic_rewards[current_without_clk_indexs] * 0
        )

        # print(-labels[with_action_indexs].float() * current_y_preds - (torch.ones(size=[len(with_action_indexs), 1]).cuda().float() - labels[with_action_indexs].float()) *
        #       (torch.ones(size=[len(with_action_indexs), 1]).cuda().float() - current_y_preds).log())
        # bce_loss = -labels[with_action_indexs].float() * current_y_preds.log() - \
        #            (torch.ones(size=[len(with_action_indexs), 1]).cuda().float() - labels[with_action_indexs].float()) * (torch.ones(size=[len(with_action_indexs), 1]).cuda().float() - current_y_preds).log()

        # if len(current_with_clk_indexs) > 0:
        # print('with clk', current_y_preds[current_with_clk_indexs], current_y_preds[current_with_clk_indexs].log())
        #     print('clk', bce_loss[current_with_clk_indexs])

        # print('without clk', current_y_preds[current_without_clk_indexs], torch.exp(-current_y_preds[current_without_clk_indexs]))

        current_basic_rewards[current_with_clk_indexs] = with_clk_rewards
        current_basic_rewards[current_without_clk_indexs] = without_clk_rewards
        # print(current_basic_rewards[current_without_clk_indexs])

        rewards[with_action_indexs, :] = current_basic_rewards

    return y_preds, rewards, return_c_actions


def test(rl_model, model_dict, embedding_layer, data_loader, device):
    targets, predicts = list(), list()
    intervals = 0
    total_test_loss = 0
    test_rewards = torch.FloatTensor().to(device)
    final_actions = torch.LongTensor().to(device)
    final_prob_weights = torch.FloatTensor().to(device)
    with torch.no_grad():
        for i, (features, labels) in enumerate(data_loader):
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)

            embedding_vectors = embedding_layer.forward(features)

            actions, c_actions, ensemble_c_actions = rl_model.choose_best_action(embedding_vectors)
            # print(actions, prob_weights)
            # print(torch.sum(prob_weights, dim=-1))
            y, rewards, return_c_actions = generate_preds(model_dict, features, actions, ensemble_c_actions, c_actions,
                                                          labels, device, mode='test')

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
            intervals += 1

            test_rewards = torch.cat([test_rewards, rewards], dim=0)

            final_actions = torch.cat([final_actions, actions], dim=0)
            final_prob_weights = torch.cat([final_prob_weights, ensemble_c_actions], dim=0)

    return roc_auc_score(targets, predicts), predicts, test_rewards.mean().item(), final_actions, final_prob_weights


def submission(rl_model, model_dict, embedding_layer, data_loader, device):
    targets, predicts = list(), list()
    final_actions = torch.LongTensor().to(device)
    final_prob_weights = torch.FloatTensor().to(device)
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)

            embedding_vectors = embedding_layer.forward(features)

            actions, c_actions, ensemble_c_actions = rl_model.choose_best_action(embedding_vectors)

            y, rewards, return_c_actions = generate_preds(model_dict, features, actions, ensemble_c_actions, c_actions,
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

def eva_stopping(valid_rewards, type, args):  # early stopping
    if len(valid_rewards) >= args.early_stop_iter:
        reward_campare_arrs = [valid_rewards[-i] < valid_rewards[-i - 1] for i in range(1, args.rl_early_stop_iter)]
        reward_div_mean = sum([abs(valid_rewards[-i] - valid_rewards[-i - 1]) for i in range(1, args.rl_early_stop_iter)]) / args.rl_early_stop_iter

        if (False not in reward_campare_arrs) or (reward_div_mean <= args.reward_epsilon):
            return True

    return False

if __name__ == '__main__':
    campaign_id = '1458/'  # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args, train_data, test_data, field_nums, feature_nums = config.init_parser(campaign_id)

    # 设置随机数种子
    setup_seed(args.seed)

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

    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.rl_gen_batch_size, num_workers=8)

    device = torch.device(args.device)  # 指定运行设备

    logger.info(campaign_id)
    logger.info('RL model ' + args.rl_model_name + ' has been training')

    ensemble_model_list = args.ensemble_models.split(',') # LR,FM,W&D,FNN,DeepFM,IPNN,OPNN,DCN,AFM

    test_predict_arrs = []

    model_dict = {}
    for i in range(len(ensemble_model_list)):
        current_model = get_ensemble_model(ensemble_model_list[i], feature_nums, field_nums, args.latent_dims).to(device)
        pretrain_params = torch.load(args.save_param_dir + args.campaign_id + ensemble_model_list[i] + 'best.pth')
        current_model.load_state_dict(pretrain_params)
        current_model.eval()
        model_dict.setdefault(i, current_model)
    model_dict_len = len(model_dict)

    # key fold training data

    gap = int(round((len(train_data) // args.rl_batch_size) // args.record_times, -1)) # 打印间隔

    train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.rl_batch_size, num_workers=8, shuffle=1)

    data_len = len(train_data)

    rl_model = get_model(model_dict_len, feature_nums, field_nums, args.latent_dims, args.init_lr_a, args.init_lr_c, data_len,
                     args.rl_batch_size,
                     args.memory_size, args.seed, device, campaign_id)

    embedding_layer = Feature_Embedding(feature_nums, field_nums, args.latent_dims).to(device)
    FM_pretrain_params = torch.load(args.save_param_dir + args.campaign_id + 'FM' + 'best.pth')
    embedding_layer.load_embedding(FM_pretrain_params)

    loss = nn.BCELoss()

    val_aucs = []

    exploration_rate = args.init_exploration_rate

    val_rewards_records = []
    timesteps = []
    train_critics = []
    global_steps = 0

    random = True
    start_time = datetime.datetime.now()

    torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

    train_start_time = datetime.datetime.now()

    is_sample_action = True # 是否在训练开始时完全使用随机动作
    is_val = False
    # for i, batchs in enumerate(get_list_data(train_data, args.rl_batch_size, True)): # 要不要早停
    #     features, labels = torch.Tensor(batchs[:, 1:]).long().to(device), torch.unsqueeze(torch.LongTensor(batchs[:, 0]), 1).to(device)
    for i, (features, labels) in enumerate(train_data_loader):
        features, labels = features.long().to(device), torch.unsqueeze(labels, 1).long().to(device)

        embedding_vectors = embedding_layer.forward(features)  # 考虑把ctr作为状态?

        c_actions, ensemble_c_actions, d_q_values, ensemble_d_actions = rl_model.choose_action(
                embedding_vectors, False if i >= gap else True)

        y_preds, rewards, return_c_actions = \
            generate_preds(model_dict, features, ensemble_d_actions, ensemble_c_actions, c_actions, labels, device,
                           mode='train')

        transitions = torch.cat(
            [features.float(), c_actions, d_q_values, ensemble_d_actions.float(), rewards],
            dim=1)
        # transitions = torch.cat(
        #     [features.float(), c_actions, d_q_values, ensemble_d_actions.float(), rewards],
        #     dim=1)

        rl_model.store_transition(transitions)

        if i % gap == 0:
            auc, predicts, test_rewards, actions, prob_weights = test(rl_model, model_dict, embedding_layer,
                                                                      test_data_loader,
                                                                      device)
            logger.info('Model {}, timesteps {}, val auc {}, val rewards {}, [{}s]'.format(
                args.rl_model_name, i * args.rl_batch_size, auc, test_rewards, (datetime.datetime.now() - train_start_time).seconds))
            val_rewards_records.append(test_rewards)
            timesteps.append(i)
            val_aucs.append(auc)

            rl_model.temprature = max(rl_model.temprature_min,
                                      rl_model.temprature_max - i *
                                      (rl_model.temprature_max - rl_model.temprature_min) / (
                                          len(train_data) // args.rl_batch_size))

            torch.cuda.empty_cache()

        if i >= gap:
            critic_loss = rl_model.learn(embedding_layer)
            train_critics.append(critic_loss)

    logger.info('Final gumbel Softmax temprature is {}'.format(rl_model.temprature))
    train_end_time = datetime.datetime.now()

    '''
    要不要早停
    '''
    test_predicts, test_auc, test_actions, test_prob_weights = submission(rl_model, model_dict,
                                                                          embedding_layer, test_data_loader,
                                                                          device)


    logger.info('Model {}, test auc {}, [{}s]'.format(args.rl_model_name,
                                                        test_auc, (datetime.datetime.now() - start_time).seconds))
    test_predict_arrs.append(test_predicts)

    prob_weights_df = pd.DataFrame(data=test_prob_weights)
    prob_weights_df.to_csv(submission_path + 'test_prob_weights' + '.csv', header=None)

    actions_df = pd.DataFrame(data=test_actions)
    actions_df.to_csv(submission_path + 'test_actions' + '.csv', header=None)

    valid_aucs_df = pd.DataFrame(data=val_aucs)
    valid_aucs_df.to_csv(submission_path + 'val_aucs' + '.csv', header=None)

    val_rewards_records = {'rewards': val_rewards_records, 'timesteps': timesteps}
    val_rewards_records_df = pd.DataFrame(data=val_rewards_records)
    val_rewards_records_df.to_csv(submission_path + 'val_reward_records' + '.csv', index=None)

    train_critics_df = pd.DataFrame(data=train_critics)
    train_critics_df.to_csv(submission_path + 'train_critics' + '.csv', header=None)

    final_subs = np.mean(test_predict_arrs, axis=0)
    final_auc = roc_auc_score(test_data[:, 0: 1].tolist(), final_subs.tolist())

    rl_ensemble_preds_df = pd.DataFrame(data=final_subs)
    rl_ensemble_preds_df.to_csv(submission_path + 'submission.csv')

    rl_ensemble_aucs = [[final_auc]]
    rl_ensemble_aucs_df = pd.DataFrame(data=rl_ensemble_aucs)
    rl_ensemble_aucs_df.to_csv(submission_path + 'ensemble_aucs.csv', header=None)

    if args.dataset_name == 'ipinyou/':
        logger.info('Dataset {}, campain {}, models {}, ensemble auc {}\n'.format(args.dataset_name,
                                                                                  args.campaign_id,
                                                                                  args.rl_model_name, final_auc))
    else:
        logger.info(
            'Dataset {}, models {}, ensemble auc {}\n'.format(args.dataset_name, args.rl_model_name, final_auc))