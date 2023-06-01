""" Center Sever - A central node with hole dataset of all clients. """

from ..FedAvg.server import Server as FedAvgServer
from utils import train_model


class Server(FedAvgServer):

    def __init__(self, center_update_samples: int = None, **kwargs):
        """
        center_update_samples: num of samples used for center node update.
            If specific, the center node's update will use center_update_samples one epoch.
        """
        super(Server, self).__init__(**kwargs)
        self.algorithm_name = "Center"

        self.center_update_samples = center_update_samples
        if center_update_samples is not None:
            self.set_train_loader(self.train_dataset, batch_size=center_update_samples)

    """ All FedAvg protocol is passed, except local update"""

    def sample_clients(self):
        """sample all clients for testing personal performance"""
        self.selected_clients = self.clients

    def distribute_model(self):
        """Still distribute model to clients for testing personal performance"""
        FedAvgServer.distribute_model(self)

    def local_update(self, epochs):
        """update server rather than clients"""
        if self.center_update_samples is None:
            # if center_update_samples not specific, all train data will be used one epoch
            FedAvgServer.update(self, epochs)
        else:
            # for fair compare, we can set center_update_samples = clients' mean_samples
            train_model(self.model, self.train_loader, self.optimizer,
                        self.loss_fn, epochs, self.device, verbose=0, one_batch=True)

    def aggregate(self):
        # do nothing
        pass
