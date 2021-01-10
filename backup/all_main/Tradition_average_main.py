import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import argparse
import random
from sklearn.metrics import roc_auc_score
import src.models.p_model as p_model
import src.models.creat_data as Data

import torch
import torch.nn as nn
import torch.utils.data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Weight_Training(nn.Module):
    def __init__(self, input_dims, weight_dims):
        super(Weight_Training, self).__init__()
        self.input_dims = input_dims
        self.weight_dims = weight_dims

        neuron_nums = [512, 256, 300]
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dims, neuron_nums[0]),
            # nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            # nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            # nn.Linear(neuron_nums[1], neuron_nums[2]),
            # nn.BatchNorm1d(neuron_nums[2]),
            # nn.ReLU(),
            nn.Linear(neuron_nums[1], self.weight_dims),
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        weights = self.mlp(input)

        return weights

def get_dataset(datapath, dataset_name, campaign_id):
    data_path = datapath + dataset_name + campaign_id
    train_data_file_name = 'train_.txt'
    train_fm = pd.read_csv(data_path + train_data_file_name, header=None).values.astype(int)

    test_data_file_name = 'test_.txt'
    test_fm = pd.read_csv(data_path + test_data_file_name, header=None).values.astype(int)

    field_nums = len(train_fm[0, 1:])  # 特征域的数量

    feature_index_name = 'featindex.txt'
    feature_index = pd.read_csv(data_path + feature_index_name, header=None).values
    feature_nums = int(feature_index[-1, 0].split('\t')[1]) + 1 # 特征数量

    train_data = train_fm
    test_data = test_fm

    return train_fm, train_data, test_data, field_nums, feature_nums


def train(model, model_dict, optimizer, data_loader, loss, device):
    model.train()  # 转换为训练模式
    total_loss = 0
    log_intervals = 0
    for i, (features, labels) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)

        pretrain_y_preds = torch.cat([
            model_dict[l](features) for l in range(len(model_dict))
        ], dim=1)

        weights = model(pretrain_y_preds)

        y = torch.sum(torch.mul(pretrain_y_preds, weights), dim=-1).view(-1, 1)

        train_loss = loss(y, labels.float())

        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()  # 取张量tensor里的标量值，如果直接返回train_loss很可能会造成GPU out of memory

        log_intervals += 1

    return total_loss / log_intervals


def test(model, model_dict, data_loader, loss, device):
    model.eval()
    targets, predicts = list(), list()
    intervals = 0
    total_test_loss = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)

            pretrain_y_preds = torch.cat([
                model_dict[l](features) for l in range(len(model_dict))
            ], dim=1)

            weights = model(pretrain_y_preds)

            y = torch.sum(torch.mul(pretrain_y_preds, weights), dim=-1).view(-1, 1)

            test_loss = loss(y, labels.float())
            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())
            intervals += 1
            total_test_loss += test_loss.item()

    return roc_auc_score(targets, predicts), total_test_loss / intervals


def submission(model, model_dict, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            pretrain_y_preds = torch.cat([
                model_dict[l](features) for l in range(len(model_dict))
            ], dim=1)

            weights = model(pretrain_y_preds)

            y = torch.sum(torch.mul(pretrain_y_preds, weights), dim=-1).view(-1, 1)

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(y.tolist())

    return predicts, roc_auc_score(targets, predicts)


def main(data_path, dataset_name, campaign_id, latent_dims, model_name, epoch, learning_rate,
         weight_decay, early_stop_type, batch_size, device, save_param_dir):
    if not os.path.exists(save_param_dir + campaign_id):
        os.mkdir(save_param_dir + campaign_id)

    device = torch.device(device)  # 指定运行设备
    train_fm, train_data, test_data, field_nums, feature_nums = get_dataset(data_path, dataset_name, campaign_id)

    train_dataset = Data.libsvm_dataset(train_data[:, 1:], train_data[:, 0])
    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    FFM = p_model.FFM(feature_nums, field_nums, latent_dims)
    FFM_pretrain_params = torch.load(save_param_dir + campaign_id + 'FFMbest.pth')
    FFM.load_state_dict(FFM_pretrain_params)
    FFM.eval()

    LR = p_model.LR(feature_nums)
    LR_pretrain_params = torch.load(save_param_dir + campaign_id + 'LRbest.pth')
    LR.load_state_dict(LR_pretrain_params)
    LR.eval()

    FM = p_model.FM(feature_nums, latent_dims)
    FM_pretrain_params = torch.load(save_param_dir + campaign_id + 'FMbest.pth')
    FM.load_state_dict(FM_pretrain_params)
    FM.eval()

    AFM = p_model.AFM(feature_nums, field_nums, latent_dims)
    AFM_pretrain_params = torch.load(save_param_dir + campaign_id + 'AFMbest.pth')
    AFM.load_state_dict(AFM_pretrain_params)
    AFM.eval()

    WandD = p_model.WideAndDeep(feature_nums, field_nums, latent_dims)
    WandD_pretrain_params = torch.load(save_param_dir + campaign_id + 'W&Dbest.pth')
    WandD.load_state_dict(WandD_pretrain_params)
    WandD.eval()

    # DeepFM = p_model.DeepFM(feature_nums, field_nums, latent_dims)
    # DeepFM_pretrain_params = torch.load(save_param_dir + campaign_id + 'DeepFMbest.pth')
    # DeepFM.load_state_dict(DeepFM_pretrain_params)
    # DeepFM.eval()

    FNN = p_model.FNN(feature_nums, field_nums, latent_dims)
    FNN_pretrain_params = torch.load(save_param_dir + campaign_id + 'FNNbest.pth')
    FNN.load_state_dict(FNN_pretrain_params)
    FNN.eval()

    IPNN = p_model.InnerPNN(feature_nums, field_nums, latent_dims)
    IPNN_pretrain_params = torch.load(save_param_dir + campaign_id + 'IPNNbest.pth')
    IPNN.load_state_dict(IPNN_pretrain_params)
    IPNN.eval()

    # OPNN = p_model.OuterPNN(feature_nums, field_nums, latent_dims)
    # OPNN_pretrain_params = torch.load(save_param_dir + campaign_id + 'OPNNbest.pth')
    # OPNN.load_state_dict(OPNN_pretrain_params)
    # OPNN.eval()

    DCN = p_model.DCN(feature_nums, field_nums, latent_dims)
    DCN_pretrain_params = torch.load(save_param_dir + campaign_id + 'DCNbest.pth')
    DCN.load_state_dict(DCN_pretrain_params)
    DCN.eval()

    # model_dict = {0: LR.to(device), 1: FM.to(device), 2: FFM.to(device)}
    model_dict = {0: WandD.to(device), 1: FNN.to(device), 2: IPNN.to(device), 3: DCN.to(device), 4: AFM.to(device),
                  5: FFM.to(device)}

    model = Weight_Training(len(model_dict), len(model_dict)).to(device)

    loss = nn.BCELoss()

    valid_aucs = []
    valid_losses = []
    early_stop_index = 0
    is_early_stop = False

    start_time = datetime.datetime.now()
    for epoch_i in range(epoch):
        torch.cuda.empty_cache()  # 清理无用的cuda中间变量缓存

        train_start_time = datetime.datetime.now()

        learning_rate += 1e-4
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_average_loss = train(model, model_dict, optimizer, train_data_loader, loss, device)

        torch.save(model.state_dict(), save_param_dir + campaign_id + model_name + str(np.mod(epoch_i, 5)) + '.pth')

        auc, valid_loss = test(model, model_dict, test_data_loader, loss, device)
        valid_aucs.append(auc)
        valid_losses.append(valid_loss)

        train_end_time = datetime.datetime.now()
        print('epoch:', epoch_i, 'training average loss:', train_average_loss, 'validation auc:', auc,
              'validation loss:', valid_loss, '[{}s]'.format((train_end_time - train_start_time).seconds))

        if eva_stopping(valid_aucs, valid_losses, early_stop_type):
            early_stop_index = np.mod(epoch_i - 4, 5)
            is_early_stop = True
            break

    end_time = datetime.datetime.now()

    if is_early_stop:
        test_model = Weight_Training(len(model_dict), len(model_dict)).to(device)
        load_path = save_param_dir + campaign_id + model_name + str(early_stop_index) + '.pth'

        test_model.load_state_dict(torch.load(load_path, map_location=device))  # 加载最优参数
    else:
        test_model = model

    auc, test_loss = test(test_model, model_dict, test_data_loader, loss, device)
    torch.save(test_model.state_dict(), save_param_dir + campaign_id + model_name + 'best.pth')  # 存储最优参数

    print('\ntest auc:', auc, datetime.datetime.now(), '[{}s]'.format((end_time - start_time).seconds))

    submission_path = data_path + dataset_name + campaign_id + model_name + '/'  # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    # 测试集submission
    test_predicts, test_auc = submission(test_model, model_dict, test_data_loader, device)
    test_pred_df = pd.DataFrame(data=test_predicts)

    test_pred_df.to_csv(submission_path + 'test_submission.csv', header=None)

    day_aucs = [[test_auc]]
    day_aucs_df = pd.DataFrame(data=day_aucs)
    day_aucs_df.to_csv(submission_path + 'day_aucs.csv', header=None)

    for i in range(5):
        os.remove(save_param_dir + campaign_id + model_name + str(i) + '.pth')


def eva_stopping(valid_aucs, valid_losses, type):  # early stopping
    if type == 'auc':
        if len(valid_aucs) >= 5:
            if valid_aucs[-1] < valid_aucs[-2] and valid_aucs[-2] < valid_aucs[-3] and valid_aucs[-3] < valid_aucs[
                -4] and valid_aucs[-4] < valid_aucs[-5]:
                return True
    else:
        if len(valid_losses) >= 5:
            if valid_losses[-1] > valid_losses[-2] and valid_losses[-2] > valid_losses[-3] and valid_losses[-3] > \
                    valid_losses[-4] and valid_losses[-4] > valid_losses[-5]:
                return True

    return False

# 用于预训练传统预测点击率模型
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi')
    parser.add_argument('--campaign_id', default='3358/', help='1458, 3358, 3386, 3427, 3476')
    parser.add_argument('--model_name', default='Tradition_avg', help='LR, FM, FFM, W&D, FNN, DeepFM, IPNN, OPNN, DCN, AFM')
    parser.add_argument('--latent_dims', default=10)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop_type', default='loss', help='auc, loss')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_param_dir', default='../models/model_params/')

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(1)

    main(
        args.data_path,
        args.dataset_name,
        args.campaign_id,
        args.latent_dims,
        args.model_name,
        args.epoch,
        args.learning_rate,
        args.weight_decay,
        args.early_stop_type,
        args.batch_size,
        args.device,
        args.save_param_dir
    )