import torch

from models.base_model import BaseModel

BATCH_LOG_STEP = 10
MODEL_LOG_STEP = 50


class ContrastiveBase(BaseModel):
    def __init__(self):
        """
        to be implemented by subclass
        """

        raise NotImplementedError

    def build_net(self, model_args):
        """
        to be implemented by subclass
        """

        raise NotImplementedError

    def build_optimizer(self, optim_args):
        params = self.encoder.parameters()

        optimizer = torch.optim.AdamW(
            params, lr=optim_args.learning_rate, weight_decay=optim_args.weight_decay
        )

        milestones = [
            int(x * optim_args.epochs) for x in optim_args.scheduler_milestones
        ]

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=optim_args.lr_gamma
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

