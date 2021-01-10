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

def get_dataset(datapath, dataset_name, campaign_id):
    data_path = datapath + dataset_name + campaign_id

    test_data_file_name = 'test_.txt'
    test_fm = pd.read_csv(data_path + test_data_file_name, header=None).values.astype(int)

    field_nums = len(test_fm[0, 1:])  # 特征域的数量

    feature_index_name = 'featindex.txt'
    feature_index = pd.read_csv(data_path + feature_index_name, header=None).values
    feature_nums = int(feature_index[-1, 0].split('\t')[1]) + 1  # 特征数量

    test_data = test_fm

    return test_data, field_nums, feature_nums


def submission(model_dict, data_loader, device):
    targets, predicts = list(), list()
    pretrain_model_len = len(model_dict)
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)

            pretrain_y_preds = torch.cat([model_dict[i](features) for i in range(pretrain_model_len)], dim=1).mean(dim=1).view(-1, 1)

            targets.extend(labels.tolist())  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            predicts.extend(pretrain_y_preds.tolist())

    return predicts, roc_auc_score(targets, predicts)


def main(data_path, dataset_name, campaign_id, latent_dims, batch_size, device, save_param_dir):
    if not os.path.exists(save_param_dir):
        os.mkdir(save_param_dir)

    device = torch.device(device)  # 指定运行设备
    test_data, field_nums, feature_nums = get_dataset(data_path, dataset_name, campaign_id)

    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    # FFM = p_model.FFM(feature_nums, field_nums, latent_dims)
    # FFM_pretrain_params = torch.load(save_param_dir + campaign_id + 'FFMbest.pth')
    # FFM.load_state_dict(FFM_pretrain_params)
    # FFM.eval()
    #
    # LR = p_model.LR(feature_nums)
    # LR_pretrain_params = torch.load(save_param_dir + campaign_id + 'LRbest.pth')
    # LR.load_state_dict(LR_pretrain_params)
    # LR.eval()

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
    model_dict = {0: WandD.to(device), 1: FNN.to(device), 2: IPNN.to(device), 3: DCN.to(device), 4: AFM.to(device)}

    submission_path = data_path + dataset_name + campaign_id + 'average_pretrain' + '/'  # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    # 测试集submission
    test_predicts, test_auc = submission(model_dict,  test_data_loader, device)

    test_pred_df = pd.DataFrame(data=test_predicts)

    test_pred_df.to_csv(submission_path + 'test_submission.csv', header=None)

    day_aucs = [[test_auc]]
    day_aucs_df = pd.DataFrame(data=day_aucs)
    day_aucs_df.to_csv(submission_path + 'day_aucs.csv', header=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, avazu')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 3386')
    parser.add_argument('--latent_dims', default=10)
    parser.add_argument('--batch_size', type=int, default=4096)
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
        args.latent_dims,
        args.batch_size,
        args.device,
        args.save_param_dir
    )