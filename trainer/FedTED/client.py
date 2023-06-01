import copy

import numpy as np

from utils.nets import TwinBranchNets
from ..FedAvg.client import Client as BaseClient
from torch.optim import *
from utils.base_train import train_model, freeze, unfreeze
import torch
import torch.nn.functional as F

from utils.loss import VanillaKDLoss


class Client(BaseClient):
    def __init__(self, distill_lr=0.001, distill_epochs=1, distill_temperature=20., **kwargs):
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

        # distill optimizer and params
        distill_optim_kwargs = copy.deepcopy(self.optim_kwargs)
        distill_optim_kwargs['lr'] = distill_lr
        self.distill_optimizer = eval(self.opt_name)(self.model.feature_extractor.parameters(), **distill_optim_kwargs)

        self.distill_epochs = distill_epochs
        self.distill_temperature = distill_temperature
        self.distill_loss_fn = VanillaKDLoss(temperature=distill_temperature)

        self.generator = None

        self.prox_z = None
        self.prox_y = None

    def update(self, epochs=1, verbose=0):
        # 1. distill feature extractor by generator
        self.distill_feature_extractor()

        # 2. decouple train feature extractor and classifier
        self.update_twin_branch(epochs)

    def distill_feature_extractor(self):
        self.model.to(self.device)
        self.model.train()
        feature_extractor = self.model.feature_extractor
        unfreeze(feature_extractor)

        for epoch in range(self.distill_epochs):
            for x, y in self.train_loader:
                if y.size(0) == 1: continue  # generator used a bn
                if self.mode == 'ec':
                    batch_z = []
                    for j in range(y.size(0)):
                        batch_z.append(self.prox_z[int(y[j])])
                    z = torch.stack(batch_z, dim=0)
                else:
                    with torch.no_grad():
                        z, _ = self.generator(y)

                x, z = x.to(self.device), z.to(self.device)

                z_ = feature_extractor(x)
                loss = self.distill_loss_fn(z_, z)

                self.distill_optimizer.zero_grad()
                if self.mode == 'ec':
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                self.distill_optimizer.step()

        self.model.to('cpu')
        torch.cuda.empty_cache()

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
                y_g = classifier_g(z) * label_prob
                y_p = classifier_p(z)

                # c. calculate loss
                loss_g = self.loss_fn(y_g, y)
                loss_p = self.loss_fn(y_p, y)
                loss = loss_g + loss_p

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
                y_g = classifier_g(z) * label_prob
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
