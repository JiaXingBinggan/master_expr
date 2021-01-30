import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import src.models.p_model as Model
import src.models.creat_data as Data

from itertools import islice

import torch
import torch.nn as nn
import torch.utils.data

import logging
import sys
import math

import threading
import time

from src.config import config as config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(model_name, feature_nums, field_nums, latent_dims):
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


def batch_generator(data, batch_size):
    lists = data
    num_batch = math.ceil(len(lists) / batch_size)

    for i in range(num_batch):
        if not i:
            np.random.shuffle(lists)
        batch_list = lists[i * batch_size: (i + 1) * batch_size]
        np.random.shuffle(batch_list)
        yield batch_list


def train(model, optimizer, data_loader, loss, device):
    model.train()  # 转换为训练模式
    total_loss = 0
    log_intervals = 0
    for i, (features, labels) in enumerate(data_loader):
        features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
        y = model(features)

        train_loss = loss(y, labels.float())

        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()  # 取张量tensor里的标量值，如果直接返回train_loss很可能会造成GPU out of memory

        log_intervals += 1

    return total_loss / log_intervals


def test(model, data_loader, loss, device):
    model.eval()
    targets, predicts = list(), list()
    intervals = 0
    total_test_loss = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)

            y = model(features)

            test_loss = loss(y, labels.float())
            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
            intervals += 1
            total_test_loss += test_loss.item()

    return roc_auc_score(targets, predicts), total_test_loss / intervals


def submission(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            y = model(features)

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())

    return predicts, roc_auc_score(targets, predicts)


def main(model, model_name, all_train_data_loader, train_data_loader, val_data_loader, test_data_loader, optimizer,
         loss, device, args):
    valid_aucs = []
    valid_losses = []
    early_stop_index = 0
    is_early_stop = False

    start_time = datetime.datetime.now()

    for epoch in range(args.epoch):
        torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

        train_start_time = datetime.datetime.now()

        train_average_loss = train(model, optimizer, train_data_loader, loss, device)

        torch.save(model.state_dict(), args.save_param_dir + args.campaign_id + model_name + str(
            np.mod(epoch, args.early_stop_iter)) + args.sample_type + '.pth')

        auc, valid_loss = test(model, val_data_loader, loss, device)
        valid_aucs.append(auc)
        valid_losses.append(valid_loss)

        train_end_time = datetime.datetime.now()
        logger.info(
            'Model {}, epoch {}, train loss {}, val auc {}, '
            'val loss {} [{}s]'.format(model_name, epoch, train_average_loss, auc, valid_loss,
                                       (train_end_time - train_start_time).seconds))

        if eva_stopping(valid_aucs, valid_losses, args.early_stop_type, args):
            early_stop_index = np.mod(epoch - args.early_stop_iter + 1, args.early_stop_iter)
            is_early_stop = True
            break

    end_time = datetime.datetime.now()

    if is_early_stop:
        test_model = get_model(model_name, feature_nums, field_nums, args.latent_dims).to(device)
        load_path = args.save_param_dir + args.campaign_id + model_name + str(
            early_stop_index) + args.sample_type + '.pth'

        test_model.load_state_dict(torch.load(load_path, map_location=device))  # 加载最优参数
    else:
        test_model = model

    test_predicts, test_auc = submission(test_model, test_data_loader, device)
    torch.save(test_model.state_dict(),
               args.save_param_dir + args.campaign_id + model_name + '_' + args.sample_type + '_best.pth')  # 存储最优参数

    logger.info('Model {}, test auc {} [{}s]'.format(model_name, test_auc,
                                                     (end_time - start_time).seconds))

    for i in range(args.early_stop_iter):
        os.remove(args.save_param_dir + args.campaign_id + model_name + str(i) + args.sample_type + '.pth')

    train_predicts, train_auc = submission(test_model, all_train_data_loader, device)

    return test_predicts, train_predicts


def eva_stopping(valid_aucs, valid_losses, type, args):  # early stopping
    if type == 'auc':
        if len(valid_aucs) >= args.early_stop_iter:
            auc_campare_arrs = [valid_aucs[-i] < valid_aucs[-i - 1] for i in range(1, args.early_stop_iter)]
            auc_div_mean = sum([abs(valid_aucs[-i] - valid_aucs[-i - 1]) for i in
                                range(1, args.early_stop_iter)]) / args.early_stop_iter

            if (False not in auc_campare_arrs) or (auc_div_mean <= args.auc_epsilon):
                return True
    else:
        if len(valid_losses) >= args.early_stop_iter:
            loss_campare_arrs = [valid_losses[-i] > valid_losses[-i - 1] for i in range(1, args.early_stop_iter)]
            loss_div_mean = sum([abs(valid_losses[-i] - valid_losses[-i - 1]) for i in
                                 range(1, args.early_stop_iter)]) / args.early_stop_iter

            if (False not in loss_campare_arrs) or (loss_div_mean <= args.loss_epsilon):
                return True

    return False


class ctrThread(threading.Thread):
    def __init__(self, func, params):
        threading.Thread.__init__(self)
        self.func = func
        self.params = params
        self.res = self.func(*self.params)

    def get_res(self):
        try:
            return self.res
        except Exception:
            return None


def get_dataset(args):
    data_path = args.data_path + args.dataset_name + args.campaign_id
    train_data_file_name = 'train.ctr.' + args.sample_type + '.txt'
    train_fm = pd.read_csv(data_path + train_data_file_name, header=None).values.astype(int)

    val_data_file_name = 'val.ctr.' + args.sample_type + '.txt'
    val_fm = pd.read_csv(data_path + val_data_file_name, header=None).values.astype(int)

    test_data_file_name = 'test.ctr.' + args.sample_type + '.txt'
    test_fm = pd.read_csv(data_path + test_data_file_name, header=None).values.astype(int)

    field_nums = train_fm.shape[1] - 1  # 特征域的数量

    with open(data_path + 'feat.ctr.' + args.sample_type + '.txt') as feat_f:
        feature_nums = int(list(islice(feat_f, 0, 1))[0].replace('\n', ''))

    train_data = train_fm
    val_data = val_fm
    test_data = test_fm

    all_train_data = np.concatenate([train_data, val_data], axis=0)

    return all_train_data, train_data, val_data, test_data, field_nums, feature_nums


# 用于预训练传统预测点击率模型
if __name__ == '__main__':
    campaign_id = '3386/'  # 1458, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)
    train_fm, train_data, val_data, test_data, field_nums, feature_nums = get_dataset(args)

    # 设置随机数种子
    setup_seed(args.seed)

    log_dirs = [args.save_log_dir, args.save_log_dir + args.campaign_id]
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.save_log_dir + str(args.campaign_id).strip('/') + '_output.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    param_dirs = [args.save_param_dir, args.save_param_dir + args.campaign_id]
    for param_dir in param_dirs:
        if not os.path.exists(param_dir):
            os.mkdir(param_dir)

    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=8)

    loss = nn.BCELoss()

    device = torch.device(args.device)  # 指定运行设备

    model_list = np.array(args.ensemble_models.split(','))
    # choose_models = model_list[np.random.choice(len(model_list), 5, replace=False)]
    choose_models = model_list
    logger.info(campaign_id)
    logger.info('Models ' + ','.join(choose_models) + ' have been trained')

    train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.test_batch_size, num_workers=8)

    val_dataset = Data.libsvm_dataset(val_data[:, 1:], val_data[:, 0])
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=8)

    train_predicts_dict = {}
    val_predicts_dict = {}
    test_predicts_dict = {}
    train_predicts_dict['label'] = train_data[:, 0].tolist()
    val_predicts_dict['label'] = val_data[:, 0].tolist()
    test_predicts_dict['label'] = test_data[:, 0].tolist()

    for model_name in choose_models:
        model = get_model(model_name, feature_nums, field_nums, args.latent_dims).to(device)
        pretrain_params = torch.load(args.save_param_dir + args.campaign_id + model_name + '_' + args.sample_type + '_best.pth')
        model.load_state_dict(pretrain_params)

        train_predicts, train_auc = submission(model, train_data_loader, device)
        val_predicts, val_auc = submission(model, val_data_loader, device)
        test_predicts, test_auc = submission(model, test_data_loader, device)
        
        train_predicts_dict.setdefault(model_name, np.array(train_predicts).flatten().tolist())
        val_predicts_dict.setdefault(model_name, np.array(val_predicts).flatten().tolist())
        test_predicts_dict.setdefault(model_name, np.array(test_predicts).flatten().tolist())
    
    train_predicts_df = pd.DataFrame(data=train_predicts_dict)
    train_predicts_df.to_csv(args.data_path + args.dataset_name + args.campaign_id
                            + 'train.rl_ctr.' + args.sample_type + '.txt', index=None)

    val_predicts_df = pd.DataFrame(data=val_predicts_dict)
    val_predicts_df.to_csv(args.data_path + args.dataset_name + args.campaign_id
                             + 'val.rl_ctr.' + args.sample_type + '.txt', index=None)

    test_predicts_df = pd.DataFrame(data=test_predicts_dict)
    test_predicts_df.to_csv(args.data_path + args.dataset_name + args.campaign_id
                             + 'test.rl_ctr.' + args.sample_type + '.txt', index=None)
        
        


        

