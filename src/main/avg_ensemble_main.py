import pandas as pd
import numpy as np
import os
import random
from sklearn.metrics import roc_auc_score
import src.models.p_model as Model
import src.models.creat_data as Data

import torch
import torch.nn as nn
import torch.utils.data
from src.config import config

import logging
import sys

from itertools import islice
np.seterr(all='raise')


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


def submission(model, data_loader, device):
    model.eval()
    predicts = list()
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.long().to(device), torch.unsqueeze(labels, 1).to(device)
            y = model(features)

            predicts.extend(y.tolist())

    return predicts

def get_dataset(args):
    data_path = args.data_path + args.dataset_name + args.campaign_id
    train_data_file_name = 'train.ctr.txt'
    train_fm = pd.read_csv(data_path + train_data_file_name, header=None).values.astype(int)

    test_data_file_name = 'test.ctr.txt'
    test_fm = pd.read_csv(data_path + test_data_file_name, header=None).values.astype(int)

    field_nums = train_fm.shape[1] - 1  # 特征域的数量

    with open(data_path + 'feat.ctr.txt') as feat_f:
        feature_nums = int(list(islice(feat_f, 0, 1))[0].replace('\n', ''))

    train_data = train_fm
    test_data = test_fm

    return train_data, test_data, field_nums, feature_nums

if __name__ == '__main__':
    campaign_id = '2259/' # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)
    train_data, test_data, field_nums, feature_nums = get_dataset(args)

    # 设置随机数种子
    setup_seed(args.seed)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.save_log_dir + str(args.campaign_id).strip('/') + '_ensemble_output.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    if not os.path.exists(args.save_param_dir + args.campaign_id):
        os.mkdir(args.save_param_dir + args.campaign_id)

    test_dataset = Data.libsvm_dataset(test_data[:, 1:], test_data[:, 0])
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

    device = torch.device(args.device)  # 指定运行设备

    choose_models = args.ensemble_models.split(',')
    logger.info(campaign_id)
    logger.info('Models ' + ','.join(choose_models) + ' have been ensembled')

    submission_path = args.data_path + args.dataset_name + args.campaign_id + 'ensemble' + '/'  # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)

    current_subs = []
    for model_name in choose_models:
        model = get_model(model_name, feature_nums, field_nums, args.latent_dims).to(device)
        pretrain_params = torch.load(args.save_param_dir + args.campaign_id + model_name + 'best.pth')
        model.load_state_dict(pretrain_params)

        current_subs.append(submission(model, test_data_loader, device))
    logger.info('ensemble auc {}'.format(roc_auc_score(test_data[:, 0: 1].tolist(), np.mean(current_subs, axis=0).tolist())))

    final_subs = np.mean(current_subs, axis=0)
    ensemble_preds_df = pd.DataFrame(data=final_subs)
    ensemble_preds_df.to_csv(submission_path + 'ensemble_' + str(args.ensemble_nums) + '_submission.csv')

    final_auc = roc_auc_score(test_data[:, 0: 1].tolist(), final_subs.tolist())
    ensemble_aucs = [[final_auc]]
    ensemble_aucs_df = pd.DataFrame(data=ensemble_aucs)
    ensemble_aucs_df.to_csv(submission_path + 'ensemble_' + str(args.ensemble_nums) + '_aucs.csv', header=None)

    if args.dataset_name == 'ipinyou/':
        logger.info('Dataset {}, campain {}, models {}, ensemble auc {}\n'.format(args.dataset_name,
                                                                             args.campaign_id, ','.join(choose_models), final_auc))
    else:
        logger.info('Dataset {}, models {}, ensemble auc {}\n'.format(args.dataset_name, ','.join(choose_models), final_auc))












