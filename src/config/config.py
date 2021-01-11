import argparse
import torch
import random
import numpy as np
import pandas as pd
from itertools import islice


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def map_f(line):
    return line.replace(' ', '').replace(':1', ',')

def get_dataset(datapath, dataset_name, campaign_id):
    data_path = datapath + dataset_name + campaign_id
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

def init_parser(campaign_id):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi, avazu')
    parser.add_argument('--campaign_id', default='2259/', help='1458, 3358, 3386, 3427, 3476, avazu')
    parser.add_argument('--model_name', default='LR', help='LR, FM, FFM, W&D, FNN, DeepFM, IPNN, OPNN, DCN, AFM')
    parser.add_argument('--latent_dims', default=10)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop_type', default='loss', help='auc, loss')
    parser.add_argument('--early_stop_iter', type=int, default=5)
    parser.add_argument('--loss_epsilon', type=float, default=1e-6)
    parser.add_argument('--auc_epsilon', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_param_dir', default='../models/model_params/')
    parser.add_argument('--save_log_dir', default='../main/logs/')
    parser.add_argument('--seed', type=int, default=1024)

    # for ensemble
    parser.add_argument('--ensemble_nums', type=int, default=3, help='3,5,7')
    parser.add_argument('--ensemble_models', default='LR,FM,FNN,IPNN,DCN')

    # for RL training
    parser.add_argument('--rl_model_name', default='Hybrid_TD3')
    parser.add_argument('--init_lr_a', type=float, default=3e-4)
    parser.add_argument('--end_lr_a', type=float, default=1e-4)
    parser.add_argument('--init_lr_c', type=float, default=3e-4)
    parser.add_argument('--end_lr_c', type=float, default=3e-4)
    parser.add_argument('--init_exploration_rate', type=float, default=1)
    parser.add_argument('--end_exploration_rate', type=float, default=0.1)
    parser.add_argument('--rl_weight_decay', type=float, default=1e-5)
    parser.add_argument('--rl_batch_size', type=int, default=128)
    parser.add_argument('--rl_iter_size', type=int, default=16)
    parser.add_argument('--rl_train_iters', type=int, default=2)
    parser.add_argument('--rl_gen_batch_size', type=int, default=4096*128)
    parser.add_argument('--memory_size', type=int, default=500000) # 感觉需要再调调

    parser.add_argument('--reward_epsilon', type=float, default=2e-2)
    parser.add_argument('--run_steps', type=float, default=1e6)
    parser.add_argument('--stop_steps', type=float, default=4e6)
    parser.add_argument('--record_times', type=int, default=100)
    parser.add_argument('--rl_early_stop_iter', type=int, default=10)

    args = parser.parse_args()
    args.campaign_id = campaign_id

    if args.ensemble_nums == 3:
        args.ensemble_models = 'LR,FM,FNN'
    elif args.ensemble_nums == 5:
        args.ensemble_models = 'LR,FM,FNN,W&D,IPNN'
    elif args.ensemble_nums == 7:
        args.ensemble_models = 'LR,FM,FNN,W&D,IPNN,DCN,AFM'

    # 设置随机数种子
    setup_seed(args.seed)

    return args


