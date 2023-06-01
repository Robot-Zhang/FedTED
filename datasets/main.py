"""
    process data (leaf or dirichlet) and save it in .json/.pt/.npy,

    '.json' same as original leaf. i.e.
    data = {
            'users':[],
            'num_samples': [],
            'user_data':[user_name:[data_list] ]
            }
    Note: in some dataset, data_list is set as path of sample.
    '.pt' is Tensor in torch
    '.npy' is numpy data
"""

import argparse

from .data import *


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="MNIST",
                        help="Name of precessing dataset,"
                             "default MNIST",
                        choices=SPLIT_SET + LEAF_SET + GEN_SET)
    parser.add_argument("--split", type=str, default="dirichlet",
                        help="split dataset type, only works when dataset is split able."
                             "default dirichlet",
                        choices=["leaf", "dirichlet", "shard"])
    parser.add_argument("--iid", type=bool, default=False,
                        help="tow situation: \n"
                             "1. *split*: split data by IID? If true, all dirichlet/shard "
                             "param will be useless.\n"
                             "2. *synthetic*: whether the generated synthetic data is IID \n"
                             "default False")
    parser.add_argument("--num_clients", "-n", type=int, default=100,
                        help="num of clients, "
                             "default 30")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="num of synthetic classes, only works when dataset=synthetic,"
                             "default 10")
    parser.add_argument("--sub_classes", "-p", nargs='+', type=int, default=None,
                        help="sub classes, e.g. [0,2,5,8,9] will only use labels with 0, 2, 5, 8, and 9,"
                             "default none")
    parser.add_argument("--dim", type=int, default=60,
                        help="num of generated X's dim, only works when dataset=synthetic, "
                             "default 60")
    parser.add_argument("--alpha", "-a", type=float, default=0.5,
                        help="1. *split*: alpha for dirichlet, control distribution of class \n"
                             "2. *synthetic*: control difference of clients' model W_k, \n"
                             "default 0.5")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="beta of synthetic generator, control difference of local data x_k,"
                             "only works when dataset=synthetic, default 0.5")
    parser.add_argument("--sigma", "-s", type=float, default=0.,
                        help="sigma for num of samples (generate by np.random.lognormal), "
                             "control balance of num_samples, default 0.")
    parser.add_argument("--shard_size", type=int, default=150,
                        help="shard size of shard split")
    parser.add_argument("--mean_samples", "-mean", type=int, default=300,
                        help="Mean number of clients' samples, default 300")
    parser.add_argument("--min_samples", "-min", type=int, default=100,
                        help="Minimum number of clients' samples, default 100")
    parser.add_argument("--max_samples", "-max", type=int, default=500,
                        help="Maximum number of clients' samples, default 500")
    parser.add_argument("--train_frac", "-tf", type=float, default=0.75,
                        help="Fraction of train set in clients' samples")
    parser.add_argument("--save_dir", type=str, default="./processed_data",
                        help="Dir for saving the processed data")
    parser.add_argument("--format", "-f", type=str, default="json",
                        help="Format of saving: pt (torch.save), npy(np.save), json(same as leaf)",
                        choices=["json", "pt", "npy"])
    parser.add_argument("--conf", type=str, default=None,
                        help="path of conf file, use yaml config dataset. "
                             "e.g. './template_conf/dirichlet.yaml'")

    args = parser.parse_args()
    # convert to OmegaConf
    conf = OmegaConf.create(vars(args))

    # read conf file and merge with args
    if conf.conf is not None:
        # conf file is main config
        print(f"Load conf file {conf.conf}")
        conf2 = OmegaConf.load(conf.conf)
        print("Merge conf file and args. (If there is difference, use conf)")
        conf = OmegaConf.merge(conf, conf2)

    # check and correct configs
    conf = check_conf(conf)
    return conf


if __name__ == "__main__":
    omega_args = parser_args()
    create(omega_args)
    print("Done!")
