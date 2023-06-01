from trainer.FedAvg.server import Server as Base_Server
from utils.nets import TwinBranchNets

class Server(Base_Server):
    def __init__(self, mode='all', heterogeneous:bool=False, **kwargs):
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
        self.algorithm_name = "FedTED-TN"

    def distribute_model(self):
        feature_extractor_w = self.model.feature_extractor.state_dict()
        classifier_w = self.model.classifier.state_dict()

        # clients' personalized classifier won't be shared anyway.
        for client in self.selected_clients:
            if not self.heterogeneous:
                client.model.feature_extractor.load_state_dict(feature_extractor_w)
            client.model.classifier.load_state_dict(classifier_w)

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
