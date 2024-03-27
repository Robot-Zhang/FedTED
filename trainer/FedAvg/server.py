""" FedAvg Sever
"""

import copy
import numpy as np
import torch
from tqdm import tqdm
from .client import Node
from collections import OrderedDict
import logging
import pandas as pd
import os

from utils import evaluate_model
import time


class Server(Node):
    def __init__(self, clients, sample_frac=0.5, metric_type='accuracy', is_record_time=False, **kwargs):

        super(Server, self).__init__(**kwargs)
        self.name = "server"
        self.init_logger()
        self.algorithm_name = "FedAvg"

        self.sample_frac = sample_frac

        self.num_clients = len(clients)
        self.clients = clients
        self.registered_client_ids = [client.id for client in clients]
        self.selected_clients_ids = []
        self.selected_clients = []

        self.glob_iter = 0
        self.total_rounds = 0

        self.metric_type = metric_type
        self.metric = OrderedDict()
        self.p_acc, self.p_loss, self.g_acc, self.g_loss = 0, 0, 0, 0

        self.time_metric = OrderedDict()
        self.is_record_time = is_record_time

    """Protocol of algorithm, in most kind of FL, this is same."""

    def run(self, rounds: int, epochs: int = 1, test_interval: int = 1, verbose=0, save_ckpt=False):
        """ The train process of FedAvg, in the child class of FedAvg server,
            override the corresponding method if the process is same.

        Args:
            rounds: total communication rounds
            epochs: local update epochs of each client
            test_interval: per x round, test performance
            verbose: 0, print nothing, 1 print acc
            save_ckpt: bool, if save ckpt in each test interval. default false.
        """
        self.log_info("")
        self.log_info("=" * 50)
        self.log_info(f"Start {rounds} rounds training by {self.algorithm_name}")

        self.total_rounds = rounds
        self.init_metric()

        for r in tqdm(range(rounds)):
            if self.is_record_time:
                round_start_time = time.time()
            # # Debug: did server w change? - before
            # server_w_before = copy.deepcopy(self.model.state_dict())
            # client0_w_before = copy.deepcopy(self.clients[0].model.state_dict())

            self.glob_iter = r

            # step 1. sample clients
            self.sample_clients()
            self.distribute_model()
            if (r + 1) % test_interval == 0:
                self.evaluate_private()

            # step 2. local update
            if self.is_record_time:
                local_start_time = time.time()

            self.local_update(epochs)

            if self.is_record_time:
                local_end_time = time.time()

            # step 3. aggregate
            self.aggregate()
            if (r + 1) % test_interval == 0:
                self.evaluate_generic()
                self.record_metric(r, self.p_acc, self.p_loss, self.g_acc, self.g_loss, verbose)

            if self.is_record_time:
                round_end_time = time.time()
                client_time = (local_end_time - local_start_time) / len(self.selected_clients)
                server_time = (round_end_time - round_start_time) - (local_end_time - local_start_time)
                self.time_metric['device'].append('clients')
                self.time_metric['time'].append(client_time)
                self.time_metric['device'].append('server')
                self.time_metric['time'].append(server_time)

            if save_ckpt:
                os.makedirs(f'./ckpt/{self.glob_iter + 1}', exist_ok=True)
                torch.save(self.model.state_dict(), f'./ckpt/{self.glob_iter + 1}/server.pth')
                for c in self.selected_clients:
                    torch.save(c.model.state_dict(), f'./ckpt/{self.glob_iter + 1}/client_{c.id}.pth')
            # # Debug: did server w change? - after
            # server_w_after = self.model.state_dict()
            # client0_w_after = self.clients[0].model.state_dict()

        # save metric
        self.save_metric()
        # close log and clear buffer
        self.close()

    """ The implementation of components in FedAvg run()"""

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randomly
        num_sampled_clients = max(int(self.sample_frac * self.num_clients), 1)
        self.selected_clients_ids = sorted(np.random.choice(self.registered_client_ids,
                                                            size=num_sampled_clients,
                                                            replace=False).tolist())
        # self.log_info(f'Selected client ids: {self.selected_clients_ids}', verbose=0)

        self.selected_clients = [self.clients[idx] for idx in self.selected_clients_ids]

        for client in self.selected_clients:
            client.glob_iter = self.glob_iter
            client.total_rounds = self.total_rounds

    def distribute_model(self):
        w = self.model.state_dict()
        for client in self.selected_clients:
            client.model.load_state_dict(w)

    def local_update(self, epochs):
        for client in self.selected_clients:
            client.update(epochs=epochs)

    def aggregate(self):
        """
        aggregate the updated and transmitted parameters from each selected client.
        """

        msg_list = [(client.num_samples, client.model.state_dict())
                    for client in self.selected_clients]
        w_dict = self.avg_weights(msg_list)
        self.model.load_state_dict(w_dict)

    @staticmethod
    def avg_weights(nk_and_wk):
        """
        n_k_and_weights: [..., (n_k, w_k), ....], where n_k is the number of samples w_k is weight.
        """
        averaged_weights = OrderedDict()

        n_sum = sum([n_k for n_k, _ in nk_and_wk])
        for i, (n_k, w_k) in enumerate(nk_and_wk):
            for key in w_k.keys():
                averaged_weights[key] = n_k / n_sum * w_k[key] if i == 0 \
                    else averaged_weights[key] + n_k / n_sum * w_k[key]
        return averaged_weights

    """Metric the performance"""

    def init_metric(self):
        self.metric['round'] = []
        self.metric[f'private_{self.metric_type}'] = []
        self.metric['private_loss'] = []
        self.metric[f'general_{self.metric_type}'] = []
        self.metric['general_loss'] = []

        if self.is_record_time:
            self.time_metric['device'] = []
            self.time_metric['time'] = []

    def record_metric(self, r, private_accuracy, private_loss, general_accuracy, general_loss, verbose):
        self.metric['round'].append(r)
        self.metric[f'private_{self.metric_type}'].append(private_accuracy)
        self.metric['private_loss'].append(private_loss)
        self.metric[f'general_{self.metric_type}'].append(general_accuracy)
        self.metric['general_loss'].append(general_loss)
        self.log_info(f"\n round {r:0>3d}, \n"
                      f"\t \t private_loss:{private_loss:.4f}, \n"
                      f"\t \t private_{self.metric_type}:{private_accuracy:.4f} \n"
                      f"\t \t general_loss:{general_loss:.4f},  \n"
                      f"\t \t general_{self.metric_type}:{general_accuracy:.4f} \n",
                      verbose=verbose)

    def evaluate_generic(self):
        acc_list, loss_list = [], []
        for client in self.selected_clients:
            g_acc, g_loss = evaluate_model(client.model, self.test_loader, self.loss_fn,
                                           self.metric_type, self.device)
            acc_list.append(g_acc)
            loss_list.append(g_loss)
        self.g_acc, self.g_loss = np.mean(acc_list), np.mean(loss_list)

    def evaluate_private(self):
        acc_list, loss_list = [], []
        for client in self.selected_clients:
            p_acc, p_loss = evaluate_model(client.model, client.test_loader, self.loss_fn,
                                           self.metric_type, self.device)
            acc_list.append(p_acc)
            loss_list.append(p_loss)
        self.p_acc, self.p_loss = np.mean(acc_list), np.mean(loss_list)

    def save_metric(self):
        pd.DataFrame(self.metric).to_csv(
            os.path.join(self.log_dir, "metric.csv"), index=False)

        if self.is_record_time:
            pd.DataFrame(self.time_metric).to_csv(
                os.path.join(self.log_dir, "time_metric.csv"), index=False)

    def close(self):
        self.receive_buff.clear()
        for c in self.clients: c.receive_buff.clear()
        logging.shutdown()
