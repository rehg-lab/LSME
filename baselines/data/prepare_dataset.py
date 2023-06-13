from email.policy import default

from data.contrastive.ABC_global import ABC as ABC_global
import torch
from torch.utils.data.dataloader import default_collate



def get_datasets_global(data_args):

    if data_args.dataset == "ABC":

        train_dataset = ABC_global(
            split="train", augmentation_file=data_args.aug_file
        )

        train_dataset_eval = ABC_global(split="train", augmentation_file="none")

        val_dataset = ABC_global(split="val")

        datasets = {
            "train_dataset": train_dataset,
            "train_dataset_eval": train_dataset_eval,
            "val_dataset": val_dataset
        }
    else:
         print("Dataset not supported")

    return datasets

def get_dataloaders(datasets, data_args):

        train_dataset = datasets["train_dataset"]
        train_dataset_eval = datasets["train_dataset_eval"]
        val_dataset = datasets["val_dataset"]
        ls_val_dataset = datasets.get("ls_val_dataset", None)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True
        )
        train_sampler_eval = torch.utils.data.SubsetRandomSampler(
            torch.randperm(len(train_dataset))[: len(val_dataset)]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=custom_collate_fn,
            batch_size=data_args.batch_size,
            num_workers=data_args.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )


        train_loader_eval = torch.utils.data.DataLoader(
            train_dataset_eval,
            collate_fn=custom_collate_fn,
            num_workers=data_args.num_workers,
            batch_size=data_args.batch_size,
            pin_memory=True,
            sampler=train_sampler_eval,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            collate_fn=custom_collate_fn,
            batch_size=data_args.batch_size,
            num_workers=data_args.num_workers,
            pin_memory=True,
        )


        train_sampler_eval_viz = torch.utils.data.SubsetRandomSampler(
            torch.randperm(len(train_dataset))[:20]
        )

        train_loader_eval_viz = torch.utils.data.DataLoader(
            train_dataset_eval,
            batch_size=1,
            sampler=train_sampler_eval_viz,
            num_workers=8,
            pin_memory=True,
        )

        val_sampler_viz = torch.utils.data.SubsetRandomSampler(
            torch.randperm(len(val_dataset))[:20]
        )

        val_loader_viz = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            sampler=val_sampler_viz,
            num_workers=8,
            pin_memory=True,
        )
        if ls_val_dataset is not None:
            ls_val_loader = torch.utils.data.DataLoader(
                ls_val_dataset, num_workers=6, batch_size=data_args.batch_size, pin_memory=True
            )
        else:
            ls_val_loader = None

        loaders = {
            "train_loader": train_loader,
            "train_loader_eval": train_loader_eval,
            "train_loader_eval_viz": train_loader_eval_viz,
            "val_loader": val_loader,
            "val_loader_viz": val_loader_viz,
            "ls_val_loader": ls_val_loader,
        }

        return loaders

def custom_collate_fn(batch):
    return default_collate(batch)
