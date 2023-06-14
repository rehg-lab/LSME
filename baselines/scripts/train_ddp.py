import torch 
import numpy as np
import random
import os
import sys
import time
import yaml
import shutil
import torch.distributed as dist

from argparse import ArgumentParser, Namespace
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta

import models
from data.prepare_dataset import (
        get_datasets_global, get_dataloaders)

torch.backends.cudnn.benchmark=True

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

def main(cfg, ckpt, cfg_p):

    ## DDP stuff
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    cfg['data_args']['batch_size'] = cfg['data_args']['batch_size'] // world_size

    model_args = Namespace(**cfg['model_args'])
    data_args = Namespace(**cfg['data_args'])
    optim_args = Namespace(**cfg['optim_args'])
    
    log_dir = os.path.join(
            cfg['exp_meta']['log_dir'],
            cfg['exp_meta']['exp_name'],
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
     
    torch.manual_seed(cfg['exp_meta']['seed'])
    np.random.seed(cfg['exp_meta']['seed'])
    random.seed(cfg['exp_meta']['seed'])
   
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=int(os.environ['WORLD_SIZE']), timeout=timedelta(seconds=5400))
    torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed_all(cfg['exp_meta']['seed'])

    print(f"Rank {local_rank + 1}/{world_size} process initialized.\n")

    model = getattr(models, model_args.model_type)(model_args, optim_args, data_args)
    
    if ckpt != '':
        print(ckpt)
        model.load(ckpt)
        log_dir = os.path.split(ckpt)[0]
        log_dir = os.path.join(log_dir + "_continued")
    
    start_epoch = model.lr_scheduler.state_dict()['last_epoch']

    if local_rank == 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        #book keeping
        shutil.copy(cfg_p, log_dir)

        #book keeping
        Tee(os.path.join(log_dir, 'log.txt'), 'a')
    else:
        writer = None
    
    if model_args.model_type in ["vispe_moco"]:
        datasets = get_datasets_global(data_args)
        loaders = get_dataloaders(datasets, data_args)
    else:
        datasets = model.get_datasets(data_args) 
        loaders = model.get_loaders(datasets, data_args)

    
    model.train_full_ddp(
            loaders, 
            writer, 
            start_epoch, 
            optim_args.epochs, 
            cfg['exp_meta']['val_freq'])

def add_arguments(parser):
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)

if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print('CKPT: =>>>> ', args.ckpt)
    
    main(cfg, args.ckpt, args.cfg)

