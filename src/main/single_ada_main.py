import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
from sklearn.metrics import roc_auc_score
import src.models.p_model as Model
from torch.autograd import Variable
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.bn_input = nn.BatchNorm1d(self.input_dims)
        self.bn_input.weight.data.fill_(1)
        self.bn_input.bias.data.zero_()

        deep_input_dims = self.input_dims
        self.layers = list()
        for neuron_num in self.neuron_nums:
            self.layers.append(nn.Linear(deep_input_dims, neuron_num))
            # self.layers.append(nn.BatchNorm1d(deep_input_dims))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=0.2))
            deep_input_dims = neuron_num

        self.layers.append(nn.Linear(deep_input_dims, self.weight_dims))
        self.layers.append(nn.Softmax(dim=-1))

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, input):
        weights = self.mlp(self.bn_input(input))

        return weights

def gumbel_softmax_sample(logits, temprature=1.0, hard=False, eps=1e-20, uniform_seed=1.0):
    U = Variable(torch.FloatTensor(*logits.shape).uniform_().cuda(), requires_grad=False)
    y = logits + -torch.log(-torch.log(U + eps) + eps)
    y = F.softmax(y / temprature, dim=-1)

    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y

    return y


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

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
        current_pretrain_y_preds, labels = current_pretrain_y_preds.float().to(device) * 1e3, torch.unsqueeze(labels, 1).to(
            device)

        weights = nn_model(current_pretrain_y_preds)
        y = torch.sum(torch.mul(current_pretrain_y_preds / 1e3, weights), dim=-1).view(-1, 1)

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
            current_pretrain_y_preds, labels = current_pretrain_y_preds.float().to(device) * 1e3, torch.unsqueeze(labels, 1).to(
                device)

            weights = nn_model(current_pretrain_y_preds)

            y = torch.sum(torch.mul(current_pretrain_y_preds / 1e3, weights), dim=-1).view(-1, 1)

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
    campaign_id = '1458/'  # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)
    args.model_name = 'S_NN_CTR'
    if campaign_id == '2259/' and args.ensemble_nums == 3:
        args.ensemble_models = 'FM,IPNN,DeepFM'

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

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=50, learning_rate=0.8)
    bdt.fit(train_data[:, 1:], train_data[:, 0])

    test_res = bdt.predict(test_data[:, 1:])
    print(roc_auc_score(test_data[:, 0], test_res))

