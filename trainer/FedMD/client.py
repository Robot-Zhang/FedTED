import copy

import torch
from torch.utils.data import TensorDataset, DataLoader
from trainer.FedAvg.client import Client as BaseClient

from utils.loss import VanillaKDLoss
from utils.base_train import train_model
from torch.optim import *


class Client(BaseClient):
    """ FedMD client """

    def __init__(self, distill_lr=0.001, distill_epochs=1, distill_temperature=20., **config):
        super(Client, self).__init__(**config)
        self.alignment_data = None
        self.glob_logits = None

        distill_optim_kwargs = copy.deepcopy(self.optim_kwargs)
        distill_optim_kwargs['lr'] = distill_lr
        self.distill_optimizer = eval(self.opt_name)(self.model.parameters(), **distill_optim_kwargs)

        self.distill_epochs = distill_epochs
        # self.temperature = distill_temperature

        # according to FedMD project, the distill_loss is MAE loss, in torch, this is L1Loss
        self.distill_loss_fn = VanillaKDLoss(temperature=distill_temperature)

        # We use this attribute to determine weather use early exit points of client model when calculate logits
        # In original FedMD, if is private mode, the logits is output from -1 layer
        # In the reimplementation in KT-pFL, the logits is not from -1 layer (from output layer instead)
        # not Implemented.

        # In vanilla FedMD, clients' models are pretrained.
        # BaseClient.update(self=self, epochs=pretrain_epoch)

    def get_logits(self):
        with torch.no_grad():
            logits = self.model(self.alignment_data)
        return logits

    def update(self, epochs=1, verbose=0):
        # 1. distill by logits
        distill_loader = DataLoader(
            TensorDataset(self.alignment_data, self.glob_logits),
            batch_size=self.batch_size
        )
        train_model(self.model, distill_loader, self.distill_optimizer,
                    self.distill_loss_fn,  self.distill_epochs, self.device, verbose)

        # 2. local update
        BaseClient.update(self, epochs, verbose)
