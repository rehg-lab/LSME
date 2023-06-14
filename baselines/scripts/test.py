import torch 
import numpy as np
import random
import os
import time
import yaml

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from data.lowshot.toys4k import BaseDataset, LowShotDataset


from data.lowshot.sampler import CategoriesSampler, CategoriesSamplerME


import models

NUM_SHOTS = 1
NUM_QUERIES = 15
NUM_WAYS = 5
NUM_EPI = 500
PRED_SEG = True
CAT = True ####### Whether this is category test or instance test
EASY = False ######### All visible objects
SHOT_GIVEN = False
CO3D = False
SINGLE_OBJ = False ####

if CO3D:
    from data.lowshot.lvis import LowShotDataset

def load_and_forward_pass(datasets, loaders, ckpt_path, model):

    if ckpt_path != '':

        state_dict = torch.load(ckpt_path, map_location='cpu')
        state_dict = state_dict['encoder_dict']
        state_dict = OrderedDict({k.replace('module.', ''): v for k,v in state_dict.items()})
        model.encoder.load_state_dict(state_dict)

    model.encoder.cuda()
    model.LS_eval_me(loaders[0], loaders[1], shot_given=SHOT_GIVEN)

def main(cfg, ckpt_p):
    model_args = Namespace(**cfg['model_args'])
    data_args = Namespace(**cfg['data_args'])
    optim_args = Namespace(**cfg['optim_args'])
        
    torch.manual_seed(cfg['exp_meta']['seed'])
    np.random.seed(cfg['exp_meta']['seed'])
    random.seed(cfg['exp_meta']['seed'])
    
    global batch_log_freq
    batch_log_freq = cfg['exp_meta']['batch_log_freq']

    model = getattr(models, model_args.model_type)(model_args, optim_args, data_args)
    base_dataset = BaseDataset(mode='global')
    ls_dataset = LowShotDataset(NUM_SHOTS, NUM_WAYS, mode='global',\
        pred_seg=PRED_SEG, category=CAT, easy=EASY) ## num shot, num way


    base_loader = torch.utils.data.DataLoader(
        base_dataset,
        num_workers=8,
        batch_size=100, 
        pin_memory=True
        )
    if not SINGLE_OBJ:
        sampler_params = [ls_dataset.all_labels_shot, ls_dataset.all_labels_query, NUM_EPI, \
            NUM_WAYS, NUM_SHOTS, NUM_QUERIES] ## num way, num shot
        ls_sampler = CategoriesSamplerME(*sampler_params)
    else:
        ####################################
        sampler_params = [ls_dataset.all_labels_query, NUM_EPI, \
            NUM_WAYS, NUM_SHOTS, NUM_QUERIES, "multi_instance_shots"] ## num way, num shot
        ls_sampler = CategoriesSampler(*sampler_params)
        #################################
    loader = torch.utils.data.DataLoader(
        ls_dataset,
        num_workers=0,
        batch_sampler=ls_sampler, 
        pin_memory=True,
        shuffle=False
        )

    datasets = [base_dataset, ls_dataset]
    loaders = [base_loader, loader]

    if os.path.isdir(ckpt_p):
        all_ckpts = sorted(os.listdir(ckpt_p))
        for ckpt in all_ckpts:
            ckpt_p_sub = os.path.join(ckpt_p, ckpt)
            print('__________________')
            print(ckpt_p_sub)

            load_and_forward_pass(datasets, loaders, ckpt_p_sub, model)
    else:
        load_and_forward_pass(datasets, loaders, ckpt_p, model)

def add_arguments(parser):
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')

if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    main(cfg, args.ckpt)
