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


class Server(Node):
    def __init__(self, clients, sample_frac=0.5, metric_type='accuracy', **kwargs):

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
            # # Debug: did server w change? - before
            # server_w_before = copy.deepcopy(self.model.state_dict())
            # client0_w_before = copy.deepcopy(self.clients[0].model.state_dict())

            self.glob_iter = r

            # step 1. sample clients
            self.sample_clients()
            self.distribute_model()

            # step 2. local update
            self.local_update(epochs)

            # step 3. aggregate
            self.aggregate()

            # test performance
            if (r + 1) % test_interval == 0:
                self.test_performance(r + 1, r + 1 == rounds, verbose, save_ckpt)

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
        self.metric['device'] = []
        self.metric['id'] = []
        self.metric['private_accuracy'] = []
        self.metric['private_loss'] = []
        self.metric['general_accuracy'] = []
        self.metric['general_loss'] = []

    def test_performance(self, r, is_final_r=False, verbose=0, save_ckpt=False):
        """Test the performance of global model and client model, include:
            generic     : accuracy, loss_value (tested by server's test_loader)
            personalized: accuracy, loss_value (tested by clients' test_loader)
        """
        # Test selected clients' model, in final round, all clients will be tested
        selected_clients = self.clients if is_final_r else self.selected_clients
        p_acc_list_c, p_loss_list_c = [], []
        acc_list_c, loss_list_c = [], []
        for client in selected_clients:
            client.eval_local_data = True
            private_accuracy, private_loss = client.evaluate(self.metric_type)
            client.eval_local_data = False
            generic_accuracy, generic_loss = evaluate_model(client.model, self.test_loader, self.loss_fn,
                                                            self.metric_type, self.device)
            p_acc_list_c.append(private_accuracy)
            p_loss_list_c.append(private_loss)
            acc_list_c.append(generic_accuracy)
            loss_list_c.append(generic_loss)
            client.log_info(f"round {r:3d}, \n"
                            f"\t private_loss:{private_loss:.4f}, \t private_accuracy:{private_accuracy:.4f} \n"
                            f"\t general_loss:{generic_loss:.4f},  \t general_accuracy:{generic_accuracy:.4f}"
                            , verbose=0)
            self.record_metric(r, 'client', client.id,
                               private_accuracy, private_loss, generic_accuracy, generic_loss)
        clients_private_accuracy, clients_private_loss = np.mean(p_acc_list_c), np.mean(p_loss_list_c)
        clients_generic_accuracy, clients_generic_loss = np.mean(acc_list_c), np.mean(loss_list_c)
        # Test server's model
        # Use mean of server model for each client as private metric
        p_acc_list, p_loss_list = [], []
        for client in self.clients:
            p_acc, p_loss = evaluate_model(self.model, client.test_loader, client.loss_fn,
                                           self.metric_type, self.device)
            p_acc_list.append(p_acc)
            p_loss_list.append(p_loss)

        server_private_accuracy, server_private_loss = np.mean(p_acc_list), np.mean(p_loss_list)
        server_generic_accuracy, server_generic_loss = self.evaluate(self.metric_type)
        self.record_metric(r,
                           'server',
                           self.id,
                           server_private_accuracy,
                           server_private_loss,
                           server_generic_accuracy,
                           server_generic_loss)

        self.log_info(f"\n round {r:0>3d}, \n"
                      "\t server:  \n"
                      f"\t \t private_loss:{server_private_loss:.4f}, \n"
                      f"\t \t private_accuracy:{server_private_accuracy:.4f} \n"
                      f"\t \t general_loss:{server_generic_loss:.4f},  \n"
                      f"\t \t general_accuracy:{server_generic_accuracy:.4f} \n"
                      f"\t clients: \n"
                      f"\t \t private_loss:{clients_private_loss:.4f}, \n"
                      f"\t \t private_accuracy:{clients_private_accuracy:.4f} \n"
                      f"\t \t general_loss:{clients_generic_loss:.4f},  \n"
                      f"\t \t general_accuracy:{clients_generic_accuracy:.4f} \n",
                      verbose=verbose)

        if save_ckpt:
            os.makedirs(f'./ckpt/{self.glob_iter + 1}', exist_ok=True)
            torch.save(self.model.state_dict(), f'./ckpt/{self.glob_iter + 1}/server.pth')
            for c in self.selected_clients:
                torch.save(c.model.state_dict(), f'./ckpt/{self.glob_iter + 1}/client_{c.id}.pth')
        # return private_accuracy, private_loss, generic_accuracy, generic_loss

    def record_metric(self, *metric_content):
        for i, (key, value) in enumerate(self.metric.items()):
            value.append(metric_content[i])

    def save_metric(self):
        pd.DataFrame(self.metric).to_csv(
            os.path.join(self.log_dir, "metric.csv"), index=False)

    def close(self):
        self.receive_buff.clear()
        for c in self.clients: c.receive_buff.clear()
        logging.shutdown()
