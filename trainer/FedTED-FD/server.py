import copy

import numpy as np
import torch
from torch.optim import *
from trainer.FedAvg.server import Server as Base_Server
from utils.nets import TwinBranchNets
from utils.loss import DiversityLoss


class Server(Base_Server):
    def __init__(self, generator, gen_epochs, gen_lr=1e-4, heterogeneous: bool = False,  mode='all', **kwargs):
        """
        Args:
            mode: mode for ablation experiment, it's values are:
                all: entire FedTED
                tw: our twin-branch network with loss in e.q. 9
        """
        super(Server, self).__init__(**kwargs)
        self.heterogeneous = heterogeneous

        # config the work mode, default is all, others for ablation experiment
        assert mode in ['all', 'tw', ], "mode is for ablation experiment"
        self.mode = mode
        for c in self.clients:
            c.mode = mode
            
        # validation the model
        assert isinstance(self.model, TwinBranchNets), \
            "FedTED need model in format of [feature_extractor, classifier]. Now, only TwinBranchNets is ok."
        self.algorithm_name = "FedTED-FD"

        self.generator = generator
        self.gen_epochs = gen_epochs
        gen_optim_kwargs = copy.deepcopy(self.optim_kwargs)
        gen_optim_kwargs['lr'] = gen_lr
        self.gen_optimizer = eval(self.opt_name)(self.generator.parameters(), **gen_optim_kwargs)

        self.diversity_loss = DiversityLoss(metric='l1')

    def distribute_model(self):
        feature_extractor_w = self.model.feature_extractor.state_dict()
        classifier_w = self.model.classifier.state_dict()

        # clients' personalized classifier won't be shared anyway.
        for client in self.selected_clients:
            if not self.heterogeneous:
                client.model.feature_extractor.load_state_dict(feature_extractor_w)
            client.model.classifier.load_state_dict(classifier_w)

            client.generator = self.generator

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

            loss = div_loss + cls_loss
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.gen_optimizer.step()

        self.generator.to('cpu')
        for cm in client_models: cm.to('cpu')
        torch.cuda.empty_cache()
