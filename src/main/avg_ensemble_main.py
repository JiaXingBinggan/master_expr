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

    test_data_file_name = 'test.rl_ctr.' + args.sample_type + '.txt'
    test_fm = pd.read_csv(data_path + test_data_file_name)

    return test_fm


if __name__ == '__main__':
    campaign_id = '1458/' # 1458, 2259, 3358, 3386, 3427, 3476, avazu
    args = config.init_parser(campaign_id)
    test_data = get_dataset(args)

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

    device = torch.device(args.device)  # 指定运行设备

    choose_models = args.ensemble_models.split(',')
    logger.info(campaign_id)
    logger.info('Models ' + ','.join(choose_models) + ' have been ensembled')

    submission_path = args.data_path + args.dataset_name + args.campaign_id + 'ensemble' + '/'  # ctr 预测结果存放文件夹位置
    if not os.path.exists(submission_path):
        os.mkdir(submission_path)
    model_subs = test_data[choose_models].values.astype(float)
    labels = test_data[['label']].values.astype(float).tolist()

    final_subs = np.mean(model_subs, axis=1).tolist()

    logger.info('ensemble auc {}'.format(roc_auc_score(labels, final_subs)))

    ensemble_preds_df = pd.DataFrame(data=final_subs)
    ensemble_preds_df.to_csv(submission_path + 'ensemble_' + str(args.ensemble_nums)
                             + '_' + args.sample_type + '_submission.csv')

    final_auc = roc_auc_score(labels, final_subs)
    ensemble_aucs = [[final_auc]]
    ensemble_aucs_df = pd.DataFrame(data=ensemble_aucs)
    ensemble_aucs_df.to_csv(submission_path + 'ensemble_' + str(args.ensemble_nums)
                            + '_' + args.sample_type + '_aucs.csv', header=None)

    if args.dataset_name == 'ipinyou/':
        logger.info('Dataset {}, campain {}, models {}, ensemble auc {}\n'.format(args.dataset_name,
                                                                             args.campaign_id, ','.join(choose_models), final_auc))
    else:
        logger.info('Dataset {}, models {}, ensemble auc {}\n'.format(args.dataset_name, ','.join(choose_models), final_auc))
