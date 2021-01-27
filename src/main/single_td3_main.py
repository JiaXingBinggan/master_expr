import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
from sklearn.metrics import roc_auc_score
import src.models.p_model as Model
import src.models.Single_TD3_model_PER as td3_model
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


def get_model(action_nums, args, device):
    RL_model = td3_model.TD3_Model(neuron_nums=args.neuron_nums,
                                   action_nums=action_nums, lr_A=args.init_lr_a, lr_C=args.init_lr_c,
                                          batch_size=args.rl_batch_size,
                                          memory_size=args.memory_size, random_seed=args.seed, device=device)

    return RL_model

def generate_preds(pretrain_y_preds, ensemble_c_actions,
                   labels, device):
    model_y_preds = torch.mul(ensemble_c_actions, pretrain_y_preds / 1e2).mean(dim=-1).view(-1, 1)
    pretrain_y_pred_means = pretrain_y_preds.mean(dim=-1).view(-1, 1)

    with_clk_indexs = (labels == 1).nonzero()[:, 0]
    without_clk_indexs = (labels == 0).nonzero()[:, 0]

    basic_rewards = torch.ones_like(labels).float()

    with_clk_rewards = torch.where(
        model_y_preds[with_clk_indexs] > pretrain_y_pred_means[with_clk_indexs],
        basic_rewards[with_clk_indexs] * 1,
        basic_rewards[with_clk_indexs] * -1
    )

    without_clk_rewards = torch.where(
        model_y_preds[without_clk_indexs] < pretrain_y_pred_means[without_clk_indexs],
        basic_rewards[without_clk_indexs] * 1,
        basic_rewards[without_clk_indexs] * -1
    )

    basic_rewards[with_clk_indexs] = with_clk_rewards
    basic_rewards[without_clk_indexs] = without_clk_rewards

    return_ctrs = torch.mul(ensemble_c_actions, pretrain_y_preds)

    return model_y_preds, basic_rewards, return_ctrs * 1e2


def test(rl_model, ensemble_nums, data_loader, device):
    targets, predicts = list(), list()
    intervals = 0
    test_rewards = torch.FloatTensor().to(device)
    final_prob_weights = torch.FloatTensor().to(device)
    with torch.no_grad():
        for i, (current_pretrain_y_preds, labels) in enumerate(data_loader):
            current_pretrain_y_preds, labels = current_pretrain_y_preds.float().to(device), torch.unsqueeze(labels, 1).to(
                device)

            s_t = torch.cat([current_pretrain_y_preds.mean(dim=-1).view(-1, 1), current_pretrain_y_preds], dim=-1)

            c_actions, ensemble_c_actions = rl_model.choose_best_action(s_t)

            y, rewards, return_ctrs = generate_preds(current_pretrain_y_preds, ensemble_c_actions, labels, device)

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
            intervals += 1

            test_rewards = torch.cat([test_rewards, rewards], dim=0)

            final_prob_weights = torch.cat([final_prob_weights, ensemble_c_actions], dim=0)

    return roc_auc_score(targets, predicts), predicts, test_rewards.mean().item(), final_prob_weights


def submission(rl_model, ensemble_nums, data_loader, device):
    targets, predicts = list(), list()
    final_prob_weights = torch.FloatTensor().to(device)
    with torch.no_grad():
        for i, (current_pretrain_y_preds, labels) in enumerate(data_loader):
            current_pretrain_y_preds, labels = current_pretrain_y_preds.float().to(device), torch.unsqueeze(labels, 1).to(device)

            s_t = torch.cat([current_pretrain_y_preds.mean(dim=-1).view(-1, 1), current_pretrain_y_preds], dim=-1)

            c_actions, ensemble_c_actions = rl_model.choose_best_action(s_t)

            y, rewards, return_ctrs = generate_preds(current_pretrain_y_preds, ensemble_c_actions, labels, device)

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())

            final_prob_weights = torch.cat([final_prob_weights, ensemble_c_actions], dim=0)

    return predicts, roc_auc_score(targets, predicts), final_prob_weights.cpu().numpy()

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
    test_data = pd.read_csv(datapath + 'test.rl_ctr.' + args.sample_type + '.txt')[columns].values.astype(float)

    return train_data * 1e2, test_data * 1e2

if __name__ == '__main__':
    campaign_id = '2259/'  # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)
    args.rl_model_name = 'S_RL_CTR'
    args.neuron_nums = [128]
    train_data, test_data = get_dataset(args)

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

    device = torch.device(args.device)  # 指定运行设备

    logger.info(campaign_id)
    logger.info('RL model ' + args.rl_model_name + ' has been training')
    logger.info(args)

    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.rl_gen_batch_size, num_workers=8)

    test_predict_arrs = []

    model_dict_len = args.ensemble_nums

    # gap = args.run_steps // args.record_times
    gap = 10000
    data_len = len(train_data)

    rl_model = get_model(model_dict_len, args, device)

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
    # 设计为每隔rl_iter_size的次数训练以及在测试集上测试一次
    # 总的来说,对于ipinyou,训练集最大308万条曝光,所以就以500万次结果后,选取连续early_stop N 轮(N轮rewards没有太大变化)中auc最高的的模型进行生成
    train_batch_gen = get_list_data(train_data, args.rl_iter_size, True)# 要不要早停
    record_list = []

    while intime_steps <= args.stop_steps:
        batchs = train_batch_gen.__next__()
        labels = torch.Tensor(batchs[:, 0: 1]).long().to(device)
        current_pretrain_y_preds = torch.Tensor(batchs[:, 1:]).float().to(device)

        s_t = torch.cat([current_pretrain_y_preds.mean(dim=-1).view(-1, 1), current_pretrain_y_preds], dim=-1)

        c_actions, ensemble_c_actions = rl_model.choose_action(
            s_t)

        y_preds, rewards, return_ctrs = generate_preds(current_pretrain_y_preds, ensemble_c_actions, labels, device)

        s_t_ = torch.cat([y_preds, return_ctrs], dim=-1)

        transitions = torch.cat(
            [s_t, c_actions, s_t_, rewards],
            dim=1)

        rl_model.store_transition(transitions)

        if intime_steps >= args.rl_batch_size:
            if intime_steps % 10 == 0:
                critic_loss = rl_model.learn()
                tmp_train_ctritics = critic_loss

        if intime_steps % gap == 0:
            auc, predicts, test_rewards, prob_weights = test(rl_model, args.ensemble_nums, test_data_loader, device)
            record_list = [auc, predicts, prob_weights]

            logger.info('Model {}, timesteps {}, val auc {}, val rewards {}, [{}s]'.format(
                args.rl_model_name, intime_steps, auc, test_rewards, (datetime.datetime.now() - train_start_time).seconds))
            val_rewards_records.append(test_rewards)
            timesteps.append(intime_steps)
            val_aucs.append(auc)

            train_critics.append(tmp_train_ctritics)

            early_aucs.append([record_param_steps, auc])
            early_rewards.append([record_param_steps, test_rewards])
            torch.save(rl_model.Actor.state_dict(),
                       args.save_param_dir + args.campaign_id + args.rl_model_name + str(
                           np.mod(record_param_steps, args.rl_early_stop_iter)) + '.pth')

            record_param_steps += 1
            if args.run_steps <= intime_steps <= args.stop_steps:
                if eva_stopping(early_rewards, args):
                    max_auc_index = sorted(early_aucs[-args.rl_early_stop_iter:], key=lambda x: x[1], reverse=True)[0][0]
                    early_stop_index = np.mod(max_auc_index, args.rl_early_stop_iter)
                    is_early_stop = True
                    break

            torch.cuda.empty_cache()

        intime_steps += batchs.shape[0]

    train_end_time = datetime.datetime.now()

    '''
    要不要早停
    '''
    # if is_early_stop:
    #     test_rl_model = get_model(model_dict_len, args.init_lr_a, args.init_lr_c,
    #                  args.rl_batch_size,
    #                  args.memory_size, args.seed, device)
    #     load_path = args.save_param_dir + args.campaign_id + args.rl_model_name + str(
    #                            early_stop_index) + '.pth'
    #
    #     test_rl_model.Actor.load_state_dict(torch.load(load_path, map_location=device))  # 加载最优参数
    # else:
    #     test_rl_model = rl_model
    #
    # test_predicts, test_auc, test_prob_weights = submission(test_rl_model, args.ensemble_nums, test_data_loader,
    #                                                                       device)

    test_auc, test_predicts, test_prob_weights = \
        record_list[0], record_list[1], record_list[2].cpu().numpy()

    # for i in range(args.rl_early_stop_iter):
    #     os.remove(args.save_param_dir + args.campaign_id + args.rl_model_name + str(i) + '.pth')

    logger.info('Model {}, test auc {}, [{}s]'.format(args.rl_model_name,
                                                        test_auc, (datetime.datetime.now() - start_time).seconds))
    test_predict_arrs.append(test_predicts)

    prob_weights_df = pd.DataFrame(data=test_prob_weights)
    prob_weights_df.to_csv(submission_path + 'test_prob_weights_' + str(args.ensemble_nums) + '_'
                           + args.sample_type + '.csv', header=None)

    valid_aucs_df = pd.DataFrame(data=val_aucs)
    valid_aucs_df.to_csv(submission_path + 'val_aucs_' + str(args.ensemble_nums) + '_'
                         + args.sample_type + '.csv', header=None)

    val_rewards_records = {'rewards': val_rewards_records, 'timesteps': timesteps}
    val_rewards_records_df = pd.DataFrame(data=val_rewards_records)
    val_rewards_records_df.to_csv(submission_path + 'val_reward_records_' + str(args.ensemble_nums) + '_'
                                  + args.sample_type + '.csv', index=None)

    train_critics_df = pd.DataFrame(data=train_critics)
    train_critics_df.to_csv(submission_path + 'train_critics_' + str(args.ensemble_nums) + '_'
                            + args.sample_type + '.csv', header=None)

    final_subs = np.mean(test_predict_arrs, axis=0)
    final_auc = roc_auc_score(test_data[:, 0: 1].tolist(), final_subs.tolist())

    rl_ensemble_preds_df = pd.DataFrame(data=final_subs)
    rl_ensemble_preds_df.to_csv(submission_path + 'submission_' + str(args.ensemble_nums) + '_'
                                + args.sample_type + '.csv')

    rl_ensemble_aucs = [[final_auc]]
    rl_ensemble_aucs_df = pd.DataFrame(data=rl_ensemble_aucs)
    rl_ensemble_aucs_df.to_csv(submission_path + 'ensemble_aucs_' + str(args.ensemble_nums) + '_'
                               + args.sample_type + '.csv', header=None)

    if args.dataset_name == 'ipinyou/':
        logger.info('Dataset {}, campain {}, models {}, ensemble auc {}\n'.format(args.dataset_name,
                                                                                  args.campaign_id,
                                                                                  args.rl_model_name, final_auc))
    else:
        logger.info(
            'Dataset {}, models {}, ensemble auc {}\n'.format(args.dataset_name, args.rl_model_name, final_auc))