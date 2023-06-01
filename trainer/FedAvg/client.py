import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import train_model, evaluate_model
import os
import logging
from torch.nn import *
from torch.optim import *


class Node:
    """ A computation node, could be clients, servers or any computed devices.

    It can be understood as driver for devices.
    """

    def __init__(self, node_id: int, model: nn.Module, train_dataset, test_dataset, num_classes,
                 batch_size=32, lr=0.001, loss: str = 'CrossEntropyLoss', opt: str = 'Adam', optim_kwargs=None,
                 log_dir: str = None, device='cpu', dataset_name=None, **kwargs):
        # basic config
        if optim_kwargs is None:
            optim_kwargs = {}
        self.id = node_id
        self.log_dir = 'log' if log_dir is None else os.path.join('log', log_dir)
        self.name = f'node_{node_id:03d}'  # this name is for find logger
        self.logger = None  # Note: before using, logger must be inited
        self.device = device
        self.dataset_name = dataset_name

        # model config: model, optimizer, criterion, and dataloader
        self.model = model

        self.lr = lr
        if optim_kwargs is None: optim_kwargs = {'lr': lr}
        self.opt_name = opt
        self.optim_kwargs = optim_kwargs
        self.optimizer = eval(opt)(self.model.parameters(), **optim_kwargs)

        self.loss_fn = eval(loss)()

        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                       shuffle=True, drop_last=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                      shuffle=True, drop_last=False)
        # update num_samples
        self.num_samples = len(train_dataset)

        # receive_buff for prototype
        self.receive_buff = []

        self.num_classes = num_classes

        self.eval_local_data = True  # if False, means eval generic testset

    def init_logger(self):
        self.logger = logging.getLogger(self.name)
        os.makedirs(self.log_dir, exist_ok=True)
        handler = logging.FileHandler(filename=os.path.join(self.log_dir, self.name) + ".log", mode='w')
        formatter = logging.Formatter(datefmt="%Y-%m-%d %H:%M:%S",
                                      fmt="%(asctime)s - %(name)s - %(levelname)-9s "
                                          "- %(filename)-8s : %(lineno)s line - %(message)s ")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_info(self, log_str, verbose=1):
        assert self.logger is not None, "logger must be inited before using."
        self.logger.info(log_str)
        if verbose == 1: print(log_str)

    def set_loader(self, train_dataset, test_dataset, batch_size=32, drop_last=False):
        """set node's train and test loader"""
        self.set_train_loader(train_dataset, batch_size, drop_last)
        self.set_test_loader(test_dataset, batch_size, drop_last)

    def set_train_loader(self, train_dataset, batch_size=32, drop_last=False):
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                       shuffle=True, drop_last=drop_last)
        # update num_samples
        self.num_samples = len(train_dataset)
        # this part is not necessary, when epoch>1, whatever drop last, all samples will be use equally
        # if drop_last: self.num_samples = self.num_samples - (self.num_samples % batch_size)

    def set_test_loader(self, test_dataset, batch_size=32, drop_last=False):
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                      shuffle=True, drop_last=drop_last)

    def update(self, epochs=1, verbose=0):
        """train node's model by local train dataset"""
        train_model(self.model, self.train_loader, self.optimizer,
                    self.loss_fn, epochs, self.device, verbose)

    def evaluate(self, metric_type='accuracy', verbose=0):
        """evaluate node's model by local test dataset
        :return correct, test_loss
        """
        return evaluate_model(self.model, self.test_loader, self.loss_fn,
                              metric_type, self.device, verbose)

    def save(self, fname='model.pt'):
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, fname))

    def load(self, fname='model.pt'):
        state_dict = torch.load(os.path.join(self.log_dir, fname))
        self.model.load_state_dict(state_dict)

    # @staticmethod
    # def get_param_size(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # eval("self.%"%self.receive_buff[method],self.receive_buff[param])
    # eg.
    def parse_msg(self):
        a = self.receive_buff
        self.receive_buff.clear()


class Client(Node):
    """FedAvg client"""

    def __init__(self, **kwargs):
        super(Client, self).__init__(**kwargs)
        self.glob_iter = 0
        self.total_rounds = 0

        # init logger of client
        self.name = f'client_{self.id:0>3d}'
        self.init_logger()
