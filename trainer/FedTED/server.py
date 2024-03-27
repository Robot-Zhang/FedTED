import copy

import numpy as np
import torch
from torch.optim import *
from trainer.FedAvg.server import Server as Base_Server
from utils.nets import TwinBranchNets
from utils.loss import DiversityLoss
import os
from utils.base_train import evaluate_model

from trainer.FedRoD.server import Server as FedRoDServer


class Server(Base_Server):
    def __init__(self, generator, gen_epochs, gen_lr=1e-4, rebuild_epochs=15, heterogeneous: bool = False, mode='ec',
                 fedted_beta=1.0, fine_tune=True, rebuild_loader=None,
                 **kwargs):
        """
        Args:
            mode: mode for ablation experiment, it's values are:
                all: entire FedTED
                tw: our twin-branch network with loss in e.q. 9
        """
        super(Server, self).__init__(**kwargs)
        self.heterogeneous = heterogeneous

        # config the work mode, default is all, others for ablation experiment
        assert mode in ['all', 'td', 'ec'], "mode is for ablation experiment"
        self.mode = mode
        for c in self.clients:
            c.mode = mode

        # validation the model
        assert isinstance(self.model, TwinBranchNets), \
            "FedTED need model in format of [feature_extractor, classifier]. Now, only TwinBranchNets is ok."
        self.algorithm_name = "FedTED"

        # generator trainer
        self.generator = generator
        self.gen_epochs = gen_epochs
        gen_optim_kwargs = copy.deepcopy(self.optim_kwargs)
        gen_optim_kwargs['lr'] = gen_lr
        self.gen_optimizer = eval(self.opt_name)(self.generator.parameters(), **gen_optim_kwargs)

        # rebuilder trainer
        self.feature_extractor = self.model.feature_extractor
        self.optimizer_fe = eval(self.opt_name)(
            filter(lambda p: p.requires_grad, self.model.feature_extractor.parameters()),
            **self.optim_kwargs)
        self.rebuild_epochs = rebuild_epochs

        self.beta = fedted_beta

        self.diversity_loss = DiversityLoss(metric='l1')

        self.mse_loss_fn = torch.nn.MSELoss()
        self.fine_tune = fine_tune

        if rebuild_loader is None:
            rebuild_loader = self.train_loader
        self.rebuild_loader = rebuild_loader

    def distribute_model(self):
        feature_extractor_w = self.model.feature_extractor.state_dict()
        classifier_w = self.model.classifier.state_dict()

        # clients' personalized classifier won't be shared anyway.
        for client in self.selected_clients:
            if not self.heterogeneous:
                client.model.feature_extractor.load_state_dict(feature_extractor_w)
            client.model.classifier.load_state_dict(classifier_w)

            if self.mode == 'ec':
                prox_z, porx_y = self.gen_prox_data()
                client.prox_z, client.porx_y = prox_z, porx_y
            else:
                client.generator = self.generator

            if self.fine_tune:
                client.fine_tune()

    def aggregate(self):
        # if not heterogeneous, aggregate feature_extractor of clients
        if not self.heterogeneous:
            msg_list = [(client.num_samples, client.model.feature_extractor.state_dict())
                        for client in self.selected_clients]
            w_dict = self.avg_weights(msg_list)

            self.model.feature_extractor.load_state_dict(w_dict)

        # aggregate clients generic classifier
        msg_list = [(client.num_samples, client.model.classifier.state_dict())
                    for client in self.selected_clients]
        w_dict = self.avg_weights(msg_list)

        self.model.classifier.load_state_dict(w_dict)

        # update generator
        self.update_generator()

        # if self.heterogeneous:
        #     # rebuild model
        #     self.rebuild_generic()
        self.rebuild_generic()

    def update_generator(self):
        # 1. init generator with device and train()
        self.generator.to(self.device)
        self.generator.train()

        # 2. get clients' classifier, set as train().
        client_models = [client.model.classifier for client in self.selected_clients]
        num_selected_clients = len(self.selected_clients)
        for cm in client_models:
            cm.eval()
            cm.to(self.device)

        # 3. train generator
        for epoch in range(self.gen_epochs):
            batch_labels = np.random.choice(self.num_classes, self.batch_size)
            y = torch.tensor(batch_labels, dtype=torch.int64).to(self.device)

            # 1. calculate diversity loss
            z, eps = self.generator(y)
            div_loss = self.diversity_loss(z, eps)

            cls_loss = 0.
            for cm in client_models:
                y_ = cm(z)
                cls_loss += self.loss_fn(y_, y)
            cls_loss /= num_selected_clients

            loss = self.beta * div_loss + cls_loss
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.gen_optimizer.step()

        self.generator.to('cpu')
        for cm in client_models: cm.to('cpu')
        torch.cuda.empty_cache()

    def rebuild_generic(self):
        """reconstruct feature extractor to get a generic model"""
        self.feature_extractor.train()
        self.feature_extractor.to(self.device)

        # get little batch of generic server or client
        x, y = next(iter(self.rebuild_loader))
        if y.size(0) <= 1:
            x, y = next(iter(self.rebuild_loader))

        # z, _ = self.generator(y)
        prox_z, prox_y = self.gen_prox_data()
        batch_z = []
        for j in range(y.size(0)):
            batch_z.append(prox_z[int(y[j])])
        z = torch.stack(batch_z, dim=0)

        for epoch in range(self.rebuild_epochs):
            x, z = x.to(self.device), z.to(self.device)

            z_ = self.feature_extractor(x)

            loss = self.mse_loss_fn(z_, z)

            self.optimizer_fe.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer_fe.step()

    def gen_prox_data(self):
        porx_z = [0.] * self.num_classes
        porx_y = list(range(self.num_classes))

        batch_labels = np.random.choice(self.num_classes, self.batch_size * 100)
        y = torch.tensor(batch_labels, dtype=torch.int64)
        z, _ = self.generator(y)

        for i in range(self.num_classes):
            idx = torch.nonzero(y == i).view(-1)
            if len(idx) > 0:
                porx_z[i] += (z[idx].sum(dim=0) / len(idx))

        return torch.stack(porx_z, dim=0), torch.tensor(porx_y)

    def evaluate_generic(self):
        # use the rebuild generic model
        self.g_acc, self.g_loss = evaluate_model(self.model, self.test_loader, self.loss_fn,
                                                 self.metric_type, self.device)

    def evaluate_private(self):
        # use the client-side model
        self.p_acc, self.p_loss = FedRoDServer.evaluate_private(self)
