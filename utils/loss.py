import torch
from torch import nn, Tensor
import torch.nn.functional as F


class VanillaKDLoss(nn.Module):
    """ According to: Distilling the Knowledge in a Neural Network,
        https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, temperature):
        super(VanillaKDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        loss = F.kl_div(F.log_softmax(student_logits / self.temperature, dim=-1),
                        F.softmax(teacher_logits / self.temperature, dim=-1),
                        reduction='batchmean') * self.temperature * self.temperature
        return loss

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

