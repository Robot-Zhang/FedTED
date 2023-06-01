import numpy as np

from utils.nets import TwinBranchNets
from ..FedAvg.client import Client as BaseClient
from torch.optim import *
from utils.base_train import train_model, freeze, unfreeze
import torch
import torch.nn.functional as F


class Client(BaseClient):
    def __init__(self, **kwargs):
        super(Client, self).__init__(**kwargs)
        # flag for work mode, will be set by server
        self.mode = 'all'
        # validation the model
        assert isinstance(self.model, TwinBranchNets), \
            "FedTED model in format of [feature_extractor, classifier]. Now, only TwinBranchNets is ok."

        # optimizer for feature_extractor
        self.optimizer_fe = eval(self.opt_name)(
            filter(lambda p: p.requires_grad, self.model.feature_extractor.parameters()),
            **self.optim_kwargs)

        # optimizer for classifiers
        unfreeze(self.model)
        freeze(self.model.feature_extractor)
        self.optimizer_cls = eval(self.opt_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()), **self.optim_kwargs)
        unfreeze(self.model)

    def update(self, epochs=1, verbose=0):
        # 1. distill feature extractor by generator
        pass
        # 2. decouple train feature extractor and classifier
        self.update_twin_branch(epochs)

    def update_twin_branch(self, epochs):
        # step 1. model init
        self.model.to(self.device)

        # to facility use
        feature_extractor = self.model.feature_extractor
        classifier_g = self.model.classifier
        classifier_p = self.model.twin_classifier

        # step 2. decouple train

        # 1. train feature extractor
        self.model.train()
        freeze(self.model)
        classifier_p.eval()
        classifier_g.eval()
        unfreeze(feature_extractor)

        for epoch in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                label_counts = self.count_labels(y)
                label_prob = torch.tensor(label_counts).to(self.device) / sum(label_counts)

                z = feature_extractor(x)
                y_g = classifier_g(z) # * label_prob
                y_p = classifier_p(z)

                # c. calculate loss
                loss_g = self.loss_fn(y_g, y)
                loss_p = self.loss_fn(y_p, y)
                loss = loss_g+loss_p

                # d. backward & step optim
                self.optimizer_fe.zero_grad()
                loss.backward()
                self.optimizer_fe.step()

        # 2. train generic and personalized branch in multitask way
        self.model.train()
        unfreeze(self.model)
        feature_extractor.eval()
        freeze(feature_extractor)
        for epoch in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                label_counts = self.count_labels(y)
                label_prob = torch.tensor(label_counts).to(self.device) / sum(label_counts)

                z = feature_extractor(x)
                y_g = classifier_g(z)  # * label_prob
                y_p = classifier_p(z)

                # c. calculate loss
                loss_g = self.loss_fn(y_g, y)
                loss_p = self.loss_fn(y_p, y)
                loss_norm = self.norm_loss_fn(classifier_g, classifier_p)
                loss = loss_g + loss_p + loss_norm

                self.optimizer_cls.zero_grad()
                loss.backward()
                self.optimizer_cls.step()

        # step 3. release gpu resource
        self.model.to('cpu')
        torch.cuda.empty_cache()

    def count_labels(self, y):
        label_counts = [0] * self.num_classes
        for i in range(self.num_classes):
            idx = torch.nonzero(y == i).view(-1)
            label_counts[i] += len(idx)
        return label_counts

    @staticmethod
    def norm_loss_fn(model1, model2):
        m1_params = [param.view(-1) for param in model1.parameters()]
        m2_params = [param.view(-1) for param in model2.parameters()]

        m1_params = torch.cat(m1_params, dim=0)
        m2_params = torch.cat(m2_params, dim=0)

        assert m1_params.size(0) == m2_params.size(0)

        loss = F.mse_loss(m1_params, m2_params)
        return loss

    def evaluate(self, metric_type='accuracy', verbose=0):
        # when evaluate personalized performance, use twin
        if self.eval_local_data:
            self.model.use_twin = True
        accuracy, loss_value = BaseClient.evaluate(self, metric_type, verbose)
        self.model.use_twin = False
        return accuracy, loss_value
