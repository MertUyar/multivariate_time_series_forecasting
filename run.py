import argparse
import os
import random
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PATHS = [_ROOT,
          os.path.join(_ROOT, 'models'),
          os.path.join(_ROOT, 'models', 'iTransformer')]
sys.path[0:0] = [p for p in _PATHS if p not in sys.path]

from exp import Exp

parser = argparse.ArgumentParser(description='GTR + iTransformer for DLAM time series forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='DLAM', help='model id')
parser.add_argument('--model', type=str, default='GTR',
                    help='label used in the setting string / checkpoint path')

# data loader
parser.add_argument('--data', type=str, default='DLAM', help='key into data_provider.data_dict')
parser.add_argument('--root_path', type=str, default='./', help='unused: the loader streams from HuggingFace')
parser.add_argument('--data_path', type=str, default='', help='unused: the loader streams from HuggingFace')
parser.add_argument('--features', type=str, default='MS',
                    help='M:multivariate->multivariate, S:univariate->univariate, MS:multivariate->univariate')
parser.add_argument('--freq', type=str, default='h', help='freq for time feature encoding')
parser.add_argument('--scale', type=int, default=1, help='1: standardise inputs, 0: raw')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')  # fixed: encoder-only model
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length of the model')
parser.add_argument('--est_horizon', type=int, default=336, help='prediction sequence length we want in total')
parser.add_argument('--do_predict', action='store_true', default=False,
                    help='after training, run predict() on validation_input and save real_prediction.npy')

# GTR
parser.add_argument('--cycle', type=int, default=24,
                    help='cycle length; also becomes cycle_len for the GTR Q table')
parser.add_argument('--period_len', type=int, default=24, help='GTR conv2d kernel period')
parser.add_argument('--use_revin', type=int, default=1, help='1: use revin, 0: no revin')
parser.add_argument('--backbone', type=str, default='iTransformer',
                    help='backbone after GTR, options: [iTransformer, DLinear]')

# model dimensions
parser.add_argument('--enc_in', type=int, default=23,
                    help='number of channels: 22 covariates + target; also sets var_num')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]; '
                         'anything other than timeF switches the loader to integer date parts')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--individual', type=int, default=0, help='DLinear individual head; True 1 False 0')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='huber', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start for OneCycleLR')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.cycle_len = args.cycle
# these are read as booleans downstream
args.scale = bool(args.scale)
args.use_revin = bool(args.use_revin)
args.individual = bool(args.individual)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

if args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else args.devices

print('Args in experiment:')
print(args)


def build_setting(ii):
    return '{}_{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_dm{}_el{}_seed{}_{}'.format(
        args.model_id,
        args.model,
        args.backbone,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.cycle,
        args.d_model,
        args.e_layers,
        fix_seed,
        ii)


if args.is_training:
    for ii in range(args.itr):
        setting = build_setting(ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, load=False)

        if args.use_gpu:
            torch.cuda.empty_cache()
else:
    ii = 0
    setting = build_setting(ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, load=True)

    if args.use_gpu:
        torch.cuda.empty_cache()
