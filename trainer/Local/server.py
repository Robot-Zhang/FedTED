""" Local Train. """
# Use center server here.
from ..Center.server import Server as CenterServer


class Server(CenterServer):

    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)
        self.algorithm_name = "Local"

    """use the run process of Center, i.e. pass all FedAvg process except metric model"""

    def distribute_model(self):
        """server don't distribute model anymore"""
        pass

    def local_update(self, epochs):
        """clients update with local data only"""
        for client in self.clients: client.update()
