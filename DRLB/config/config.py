import argparse
import torch
import random
import numpy as np
import pandas as pd
import datetime
import sys

from itertools import islice
from RLIB.models import p_model as Model
from RLIB.models.Feature_embedding import Feature_Embedding

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_parser(campaign_id):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi, avazu')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 3358, 3386, 3427, 3476, avazu')
    parser.add_argument('--ctr_model_name', default='LR', help='LR,FM,FNN...')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--model_name', default='DRLB')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--neuron_nums', type=list, default=[100, 100, 100])
    parser.add_argument('--memory_size', type=float, default=100000)
    parser.add_argument('--rl_batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_param_dir', default='../models/model_params/')
    parser.add_argument('--save_log_dir', default='logs/')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--sample_type', default='all', help='all, down, rand')

    parser.add_argument('--budget', type=float, default=16e6)
    parser.add_argument('--budget_para', type=list, default=[1/1], help='1,2,4')

    # op 缩放，nop 不缩放，clk

    args = parser.parse_args()
    args.campaign_id = campaign_id

    return args