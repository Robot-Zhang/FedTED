from utils.nets import TwinBranchNets
from ..FedAvg.client import Client as BaseClient
from torch.optim import *
from utils.base_train import train_model, freeze, unfreeze
import torch


class Client(BaseClient):
    def __init__(self, **kwargs):
        super(Client, self).__init__(**kwargs)

        assert isinstance(self.model, TwinBranchNets), \
            "FedRod need model in format of [feature_extractor, classifier]. Now, only TwinBranchNets is ok."

        # pointer for feature_extractor, personalized head and generic head.
        # reference to the paper.
        self.feature_extractor = self.model.feature_extractor
        self.classifier_g = self.model.classifier
        self.classifier_p = self.model.twin_classifier

        # optimizer for feature_extractor and classifier
        freeze(self.classifier_p)
        self.opt_g = eval(self.opt_name)(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         **self.optim_kwargs)
        unfreeze(self.classifier_p)
        self.opt_p = eval(self.opt_name)(self.classifier_p.parameters(),
                                         **self.optim_kwargs)

        # get label counts for cal br loss as e.q. 5 in paper
        self.label_counts = [0] * self.num_classes
        target_transform = self.train_dataset.target_transform
        for _, target in self.train_dataset:
            self.label_counts[int(target_transform(target))] += 1

    """Local update of FedRod, use br loss for generic update, use empirical loss for personalized head"""

    def update(self, epochs=1, verbose=0):
        # generic update
        freeze(self.classifier_p)
        train_model(self.model, self.train_loader, self.opt_g,
                    self.br_loss_fn, epochs, self.device, verbose)

        # personalized update
        freeze(self.model)
        self.model.use_twin = True
        unfreeze(self.classifier_p)
        train_model(self.model, self.train_loader, self.opt_g,
                    self.loss_fn, epochs, self.device, verbose)

        unfreeze(self.model)
        self.model.use_twin = False

    def br_loss_fn(self, log_probs, labels):
        # Eq. 5 in FedRoD, here, we take $q_{y_i} $ as \frac{1}{N_{m,{y_i}}}
        br_loss_values = []
        for i in range(self.num_classes):
            idx = torch.nonzero(labels == i).view(-1)
            if len(idx) > 0:
                labels_i = labels[idx]
                log_probs_i = log_probs[idx]
                l_em = self.loss_fn(log_probs_i, labels_i)
                br_loss_values.append(l_em/self.label_counts[i])
        loss = sum(br_loss_values)
        return loss
