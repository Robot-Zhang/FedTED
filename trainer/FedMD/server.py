import copy
from tqdm import tqdm
from trainer.FedAvg.server import Server as BaseServer
import numpy as np
from utils.base_train import train_model
from torch.utils.data import DataLoader, Dataset
import torch


class Server(BaseServer):
    """FedMD Server
        num_alignment: size of public for distilling. Here we ignore this arg.
        According to the FedMD paper, the number of alignment is set as 5000"""

    def __init__(self, public_dataset: Dataset, pretrain_epochs=25, num_alignment=5000, **config):
        super(Server, self).__init__(**config)
        self.algorithm_name = "FedMD"

        self.public_dataset = public_dataset

        self.alignment_loader = DataLoader(self.public_dataset, batch_size=num_alignment, shuffle=True)

        self.alignment_data = None
        self.logits = None

        # pre-train clients' model in public dataset.
        if pretrain_epochs > 0:
            print(f"Pretrain clients on public dataset, epochs={pretrain_epochs}")
            for client in tqdm(self.clients):
                # if shakespeare, embedding is not for pretrain
                # if self.dataset_name == 'shakespeare':
                #     client.model.feature_extractor.use_embedding = False
                train_model(client.model,
                            DataLoader(self.public_dataset, batch_size=client.batch_size),
                            client.optimizer,
                            client.loss_fn,
                            epochs=pretrain_epochs,
                            device=client.device,
                            verbose=0)
                torch.cuda.empty_cache()
                # if self.dataset_name == 'shakespeare':
                #     client.model.feature_extractor.use_embedding = True

    def distribute_model(self):
        """override the distribute_model func in FedAvg as
            distribute alignment data and logits to clients
        """
        # distribute new alignment_data
        self.alignment_data, _ = next(iter(self.alignment_loader))
        for client in self.selected_clients:
            client.alignment_data = self.alignment_data

        # get clients' logits and avg it
        clients_logits = [client.get_logits() for client in self.selected_clients]
        self.logits = sum(clients_logits) / len(clients_logits)

        # sent averaged logits to clients
        for client in self.selected_clients:
            client.glob_logits = self.logits

    def aggregate(self):
        """do nothing"""
        pass
