"""
FedAvg

The implementation of:
    H. Brendan McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas,
    “Communication-efficient learning of deep networks from decentralized data,”
    Proc. 20th Int. Conf. Artif. Intell. Stat. AISTATS 2017, 2017. https://arxiv.org/abs/1602.05629

"""
from .client import Client
from .server import Server
