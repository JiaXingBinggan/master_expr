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

class Weight_Training(nn.Module):
    def __init__(self, input_dims, weight_dims, neuron_nums):
        super(Weight_Training, self).__init__()
        self.input_dims = input_dims
        self.weight_dims = weight_dims
        self.neuron_nums = neuron_nums

        deep_input_dims = self.input_dims
        self.layers = list()
        for neuron_num in self.neuron_nums:
            self.layers.append(nn.Linear(deep_input_dims, neuron_num))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        self.layers.append(nn.Linear(deep_input_dims, self.weight_dims))
        self.layers.append(nn.Softmax(dim=-1))

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, input):
        weights = self.mlp(input)

        return weights

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(args, device):
    nn_model = Weight_Training(
        input_dims=args.ensemble_nums,
        weight_dims=args.ensemble_nums,
        neuron_nums=args.neuron_nums).to(device)

    return nn_model

def train(nn_model, loss, data_loader, optimizer, device):
    intervals = 0
    total_train_loss = 0
    nn_model.train()

    for i, (current_pretrain_y_preds, labels) in enumerate(data_loader):
        current_pretrain_y_preds, labels = current_pretrain_y_preds.float().to(device), torch.unsqueeze(labels, 1).to(
            device)

        weights = nn_model(current_pretrain_y_preds)
        y = torch.sum(torch.mul(current_pretrain_y_preds, weights), dim=-1).view(-1, 1)

        train_loss = loss(y, labels.float())
        nn_model.zero_grad()
        train_loss.backward()
        optimizer.step()

        intervals += 1
        total_train_loss += train_loss.item()

    return total_train_loss / intervals


def test(nn_model, loss, data_loader, device):
    targets, predicts = list(), list()
    intervals = 0
    total_test_loss = 0
    final_weights = torch.FloatTensor().to(device)
    nn_model.eval()

    with torch.no_grad():
        for i, (current_pretrain_y_preds, labels) in enumerate(data_loader):
            current_pretrain_y_preds, labels = current_pretrain_y_preds.float().to(device), torch.unsqueeze(labels, 1).to(
                device)

            weights = nn_model(current_pretrain_y_preds)

            y = torch.sum(torch.mul(current_pretrain_y_preds, weights), dim=-1).view(-1, 1)

            test_loss = loss(y, labels.float())
            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
            intervals += 1
            total_test_loss += test_loss.item()

            final_weights = torch.cat([final_weights, weights], dim=0)

    return roc_auc_score(targets, predicts), predicts, total_test_loss / intervals, final_weights

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

    return train_data, test_data

if __name__ == '__main__':
    campaign_id = '2259/'  # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)
    args.model_name = 'S_NN_CTR'
    train_data, test_data = get_dataset(args)

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
    logger.info('NN model ' + args.model_name + ' has been training')
    logger.info(args)

    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.rl_gen_batch_size, num_workers=8)

    test_predict_arrs = []
    
    nn_model = get_model(args, device)
    loss = nn.BCELoss()

    optimizer = torch.optim.Adam(params=nn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_losses = []

    start_time = datetime.datetime.now()

    torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

    train_start_time = datetime.datetime.now()

    train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)
    record_list = []

    for epoch in range(args.epoch):
        torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

        train_average_loss = train(nn_model, loss, train_data_loader, optimizer, device)

        auc, predicts, valid_loss, final_weights = \
            test(nn_model, loss, test_data_loader, device)
        record_list = [auc, predicts, final_weights]

        train_end_time = datetime.datetime.now()
        logger.info(
            'Model {}, epoch {}, train loss {}, val auc {}, '
            'val loss {} [{}s]'.format(args.model_name, epoch, train_average_loss, auc, valid_loss,
                                       (train_end_time - train_start_time).seconds))
        train_losses.append(train_average_loss)

    test_auc, test_predicts, test_prob_weights = \
        record_list[0], record_list[1], record_list[2].cpu().numpy()

    train_end_time = datetime.datetime.now()

    logger.info('Model {}, test auc {}, [{}s]'.format(args.model_name,
                                                        test_auc, (datetime.datetime.now() - start_time).seconds))
    test_predict_arrs.append(test_predicts)

    prob_weights_df = pd.DataFrame(data=test_prob_weights)
    prob_weights_df.to_csv(submission_path + 'test_prob_weights_' + str(args.ensemble_nums) + '_'
                           + args.sample_type + '.csv', header=None)

    train_critics_df = pd.DataFrame(data=train_losses)
    train_critics_df.to_csv(submission_path + 'train_losses_' + str(args.ensemble_nums) + '_'
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
                                                                                  args.model_name, final_auc))
    else:
        logger.info(
            'Dataset {}, models {}, ensemble auc {}\n'.format(args.dataset_name, args.model_name, final_auc))