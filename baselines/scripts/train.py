import torch
import numpy as np
import random
import os
import yaml
import shutil
import sys

from argparse import ArgumentParser, Namespace
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from data.prepare_dataset import (
        get_datasets_local, get_datasets_local_sv,
        get_datasets_global, get_dataloaders, get_dataloaders_non_distributed)

import models

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True


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
    model_args = Namespace(**cfg["model_args"])
    data_args = Namespace(**cfg["data_args"])
    optim_args = Namespace(**cfg["optim_args"])

    log_dir = os.path.join(
        cfg["exp_meta"]["log_dir"],
        cfg["exp_meta"]["exp_name"],
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )

    if ckpt != "":
        log_dir = os.path.split(ckpt)[0]
        log_dir = os.path.join(log_dir + "_continued")

    # tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # book keeping
    Tee(os.path.join(log_dir, "log.txt"), "a")
    shutil.copy(cfg_p, log_dir)

    torch.manual_seed(cfg["exp_meta"]["seed"])
    np.random.seed(cfg["exp_meta"]["seed"])
    random.seed(cfg["exp_meta"]["seed"])

    model = getattr(models, model_args.model_type)(model_args, optim_args, data_args)
    print(model_args.model_type)

    if ckpt != "":
        model.load(ckpt)


    start_epoch = model.lr_scheduler.state_dict()["last_epoch"]
    print('Start Epoch: ', start_epoch)

    datasets = get_datasets_local(data_args)
    loaders = get_dataloaders_non_distributed(datasets, data_args)

    model.train_full(
        loaders, writer, start_epoch, optim_args.epochs, cfg["exp_meta"]["val_freq"]
    )


def add_arguments(parser):
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")


if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg, args.ckpt, args.cfg)
