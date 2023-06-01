import os
import random
import json
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig
from matplotlib import pyplot as plt
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset
from .torchvision_dataset import get_dataset_by_name

from datasets.transforms.my_transforms import get_transform

from datasets.raw.Leaf.sent140.glove6B300d.get_mini_embs import mini_embs

NUM_CLASS = {"MNIST": 10, "FashionMNIST": 10, "EMNIST": 47, "CIFAR10": 10, "CIFAR100": 100, "femnist": 62, "celeba": 2,
             'sent140': 2, 'shakespeare': 80}
SPLIT_SET = ["MNIST", "FashionMNIST", "EMNIST", "CIFAR10", "CIFAR100"]
LEAF_SET = ["femnist", "celeba", "reddit", "sent140", "shakespeare"]
GEN_SET = ["synthetic"]
DEFAULT_CONF_PATH = "datasets/template_conf/default.yaml"
LEAF_PATH = "datasets/raw/Leaf"

"""1. functions for conf file and args"""


def check_conf(conf):
    """check kwargs and correct part of it."""
    # check legality of args
    assert 0 < conf.train_frac < 1
    assert conf.mean_samples > 0
    assert conf.max_samples > conf.mean_samples > conf.min_samples
    if conf.dataset in GEN_SET:
        assert conf.sub_classes is None, f"sub_classes should be set as None when dataset is {conf.dataset}"

    # get implied args
    if conf.sub_classes is None:
        if conf.dataset in SPLIT_SET: conf.num_classes = NUM_CLASS[conf.dataset]
        conf.sub_classes = list(range(conf.num_classes))
    else:
        conf.num_classes = len(conf.sub_classes)
    if conf.iid: conf.split = 'shard'
    conf.dataset_name = conf.dataset
    return conf


"""2. functions for create datasets """


# main function

def create(conf: DictConfig, verbose=1):
    """create data according to conf.yaml file and save it"""
    # 1. load data
    if verbose == 1:
        print(f"Load dataset {conf.dataset_name}")
        print(f"Processing ...")

    if conf.dataset_name in LEAF_SET:
        # read the processed leaf data directly
        path = os.path.join(LEAF_PATH, conf.dataset)
        with open(os.path.join(path, 'train.json'), 'r') as inf:
            train_data = json.load(inf)
        with open(os.path.join(path, 'test.json'), 'r') as inf:
            test_data = json.load(inf)
        # subset according to num_clients, min_samples, and max_samples
        train_data, test_data = sub_leaf(conf.num_clients, conf.min_samples, conf.max_samples,
                                         train_data, test_data)
    elif conf.dataset_name in SPLIT_SET:
        # split torchvision datasets, process as dict:
        # {'users':[], 'num_samples': [], 'user_data':[user_name:[data_list]]}
        train_data, test_data = split_dataset(**conf)
        # plot split image for validation
        if verbose == 1:  polt_data(train_data, 5, 5)
    elif conf.dataset_name in GEN_SET:
        # synthetic dataset with leaf formats
        train_data, test_data = synthetic(**conf)
        pass
    else:
        raise NotImplementedError

    # 2. format trans
    if verbose == 1: print(f"Converting format as {conf.format}")
    train_data, test_data = format_as(train_data, conf.format), format_as(test_data, conf.format)

    # 3. save data
    # make save dir
    os.makedirs(conf.save_dir, exist_ok=True)

    # get file name and save path
    f_name = get_f_name(**conf) + '.' + conf.format
    train_path = os.path.join(conf.save_dir, "train-" + f_name)
    test_path = os.path.join(conf.save_dir, "test-" + f_name)

    # save
    if verbose == 1:
        print(f"Saving Data to \n "
              f"\t Dir : {conf.save_dir} \n "
              f"\t File name: train(test)-{f_name}")
    if conf.format == 'json':
        with open(train_path, 'w') as outfile:
            json.dump(train_data, outfile)
        with open(test_path, 'w') as outfile:
            json.dump(test_data, outfile)
    if conf.format == 'pt':
        torch.save(train_data, train_path)
        torch.save(test_data, test_path)
    if conf.format == 'npy':
        np.save(train_path, train_data)
        np.save(test_path, test_data)

    # if dataset is sent 140, get a mini embs
    if conf.dataset == 'sent140':
        mini_embs(data_dir=conf.save_dir,
                  files=[f'train-{f_name}', f'test-{f_name}'],
                  embs_path=os.path.join(LEAF_PATH, 'sent140', 'glove6B300d', 'embs.json'),
                  target_dir=conf.save_dir,
                  fname='sent140-mini-embs')


# main create functions

def sub_leaf(num_clients, min_samples, max_samples, *old_data_tuple):
    """get subset of leaf data

    :param
        num_clients: num_clients of subset
        min_samples: clients' min_samples in subset
        max_samples: clients' max_samples in subset
        old_data_tuple: train_data, test_data, or val_data. Note, need leaf structure
    :return
        sub_data_tuple: same sequence as input old_data_tuple. e.g.
        '''train_data, test_data = get_sub_leaf(num_clients, min_samples, max_samples, train_data, test_data)'''
    """
    clients = old_data_tuple[0]['users']
    print(f"{len(clients)} clients loaded.")

    # delete illegal num_sample clients
    remove_clients = []
    for c in clients:
        c_num_sample = len(old_data_tuple[0]['user_data'][c]['y']) + \
                       len(old_data_tuple[1]['user_data'][c]['y'])
        if c_num_sample < min_samples or c_num_sample > max_samples: remove_clients.append(c)
    for c in remove_clients: clients.remove(c)
    print(f"Delete clients num: {len(remove_clients)},\n"
          f"\t rule: num_samples in range [{min_samples}, {max_samples}]\n"
          f"Rest clients num: {len(clients)}.")

    # check num
    assert len(clients) > num_clients, "Value of num_clients or min_samples is too large!"

    # select clients and their data
    selected_clients = random.sample(clients, num_clients)
    sub_data_tuple = tuple([defaultdict() for i in range(len(old_data_tuple))])
    for i, data in enumerate(old_data_tuple):
        sub_user_data = defaultdict()
        sub_data_tuple[i]['users'] = selected_clients
        sub_data_tuple[i]['num_samples'] = []
        for c in selected_clients:
            sub_user_data[c] = data['user_data'][c]
            sub_data_tuple[i]['num_samples'].append(len(sub_user_data[c]['y']))
        sub_data_tuple[i]['user_data'] = sub_user_data

    return sub_data_tuple


def split_dataset(dataset_name: str, num_clients: int, num_classes: int, alpha: float, sigma: float,
                  mean_samples: int, min_samples: int, max_samples: int, sub_classes: [int],
                  train_frac: float, split: str, iid: bool, shard_size: int, **kwargs):
    """Split torch_vision dataset by Dirichlet or Shard.

    The Dirichlet split is reference to:
        `Federated Learning Based on Dynamic Regularization, ICLR 2021, https://arxiv.org/pdf/2111.04263.pdf`.
    The Shard split is reference to:
        H. Brendan McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas,
        “Communication-efficient learning of deep networks from decentralized data,”
        Proc. 20th Int. Conf. Artif. Intell. Stat. AISTATS 2017, 2017. https://arxiv.org/abs/1602.05629


    :return
        train_data: leaf format train dataset
        test_data: leaf format test dataset, i.e.
        {'users':[],
        'num_samples': [],
        'user_data':{'user_1':{'x':[],'y':[]},
                     'user_2':{'x':[],'y':[]},
                    }
        }
    """

    assert dataset_name in SPLIT_SET, f"{dataset_name} cannot be split by {split}, use '--split leaf' instead"

    # Step 1. get raw dataset from torchvision.dataset
    dataset = get_dataset_by_name(dataset_name, is_train=True)
    if isinstance(dataset.targets, list): dataset.targets = np.array(dataset.targets)
    if isinstance(dataset.targets, torch.Tensor): dataset.targets = dataset.targets.numpy()
    if isinstance(dataset.targets, torch.Tensor): dataset.data = dataset.data.numpy()
    user_names = [f'client_{i:d}' for i in range(num_clients)]

    # Step 2. split dtasets by dirichlet or shard
    # (1) get num_samples for each client by lognorm
    num_samples = get_num_samples(num_clients, sigma,
                                  mean_samples, min_samples, max_samples)

    # (2) assign num of class data by dirichlet or shard
    # init cls_sample_matrix, structure as:
    #           client 1    client j    client m
    # cls 1        9          ...          ...
    # cls i        0          ...          ...
    # cls n        8          ...          ...
    #           sum() = num_sample
    cls_sample_matrix = np.zeros((num_clients, num_classes), dtype=int)
    if split == 'dirichlet' and not iid:
        # a. get dirchlet prob
        dirichlet_prob = np.random.dirichlet(alpha=[alpha] * num_classes,
                                             size=num_clients)
        cum_prob = np.cumsum(dirichlet_prob, axis=1)
        # b. update cls_sample_matrix by dirchlet prob
        for i in range(num_clients):
            for j in range(num_samples[i]):
                idx = np.where(cum_prob[i] > np.random.rand())[0][0]
                cls_sample_matrix[i][idx] += 1
    elif split == 'shard' and not iid:
        # a. reassign num_shards and num_samples of each client
        num_shards = (num_samples / shard_size).astype(int)
        num_samples = num_shards * shard_size
        # b. init temp vars
        assigned_num = [0] * num_classes  # record how many data assigned
        total_num_cls = [np.argwhere(dataset.targets.astype(int) == cls).reshape(-1).shape[0]
                         for cls in sub_classes]  # get num of total data
        rest_c = list(range(num_classes))  # select data from here
        # c. check legality
        assert sum(total_num_cls) >= sum(num_samples), "'num_samples' or 'num_clients' too large, set lower and rerun!"
        # d. update cls_sample_matrix by shard
        for i in range(num_clients):
            # assign one shard to client i when not done
            while np.sum(cls_sample_matrix, axis=1)[i] < num_samples[i]:
                assign_size = shard_size
                while assign_size > 0:
                    # random get shard cls
                    c = random.choice(rest_c)
                    # decide take home many samples of cls
                    num_take = assign_size if total_num_cls[c] - assigned_num[c] > assign_size \
                        else total_num_cls[c] - assigned_num[c]
                    # take num_take samples
                    cls_sample_matrix[i][c] += num_take
                    assigned_num[c] += num_take
                    # update assign_size and rest_c
                    assign_size -= num_take
                    if total_num_cls[c] == assigned_num[c]: rest_c.remove(c)
    elif iid:
        # let the cls be equal
        for i in range(num_clients):
            cls_sample_matrix[i] = (cls_sample_matrix[i] + 1) * int(num_samples[i] / num_classes)
    else:
        raise NotImplementedError

    # (3) Split dataset by classes' indices
    # a. init the sample index (original dataset) of clients
    sample_indices_list = [np.array([], dtype=int)] * num_clients
    # b. get the sample index range of each client [start_0, end_0/start_1, end_1/start_2, ...]
    cumsum_cls_num = np.cumsum(cls_sample_matrix, axis=0).transpose()
    start_end_idxs = [[0] + cumsum_cls_num[i].tolist() for i in range(num_classes)]
    # c. find corresponding indies of each cls, assign to clients.
    for c, cls in enumerate(sub_classes):
        # find indices of cls in original
        indices = np.argwhere(dataset.targets.astype(int) == cls).reshape(-1)
        # check if end_-1 out of range
        assert indices.shape[0] >= start_end_idxs[c][-1], \
            "**** ARGS ERROR **** : 'num_samples' or 'num_clients' too large, set lower and rerun!"
        # assign cls indices to clients
        for i in range(num_clients):
            sample_indices_list[i] = np.concatenate(
                (sample_indices_list[i], indices[start_end_idxs[c][i]: start_end_idxs[c][i + 1]]))

    # (4) get data & targets of each client
    clients_all_data = [{'x': dataset.data[indices].squeeze(), 'y': dataset.targets[indices].squeeze()}
                        for indices in sample_indices_list]
    # map old class into new subclass, this part is typically use in FedMD:
    #   'FedMD: Heterogenous Federated Learning via Model Distillation,
    #    NeurIPS 2019, http://arxiv.org/abs/1910.03581'
    for data in clients_all_data:
        for new_cls, cls in enumerate(sub_classes):
            data['y'] = np.where(data['y'] == cls, new_cls, data['y'])
            data['y'] = data['y'].astype(int)

    # (5) divide into train and test datasets
    clients_train_data, clients_test_data = {}, {}
    train_num_samples, test_num_samples = [], []
    for i, data in enumerate(clients_all_data):
        # shuffle data
        rand_idx = np.random.permutation(int(data['y'].shape[0]))
        data['x'] = data['x'][rand_idx]
        data['y'] = data['y'][rand_idx]
        # split train and test
        idx = int(data['y'].shape[0] * train_frac)
        train_x, train_y = data['x'][:idx].tolist(), data['y'][:idx].tolist()
        test_x, test_y = data['x'][idx:].tolist(), data['y'][idx:].tolist()
        clients_train_data[user_names[i]] = {'x': train_x, 'y': train_y}
        clients_test_data[user_names[i]] = {'x': test_x, 'y': test_y}
        train_num_samples.append(len(train_y))
        test_num_samples.append(len(test_y))

    test_data = {'users': user_names, 'num_samples': test_num_samples, 'user_data': clients_test_data}
    train_data = {'users': user_names, 'num_samples': train_num_samples, 'user_data': clients_train_data}
    return train_data, test_data


def synthetic(iid: bool, num_clients: int, num_classes: int, dim: int,
              alpha: float, beta: float, sigma: float, mean_samples: int, min_samples: int,
              max_samples: int, train_frac: float, **kwargs):
    """synthetic dataset

    Synthetic is first proposed in:
        Shamir, O., Srebro, N., and Zhang, T. Communication efficient distributed optimization
        using an approximate newton-type method. In International Conference on Machine Learning, 2014.

    This implementation is reference to [FedProx](https://github.com/litian96/FedProx.git)

    For details, recommend to read:
        Federated optimization in heterogeneous networks, MLSys 2020, https://arxiv.org/abs/1812.06127
        Page 7: Synthetic data

    In short, genergate by:
    .. math::
        y = argmax(softmax(W x + b)),
    for client k, if num_classes=10, dim=60, then:
    .. math::
        W_k ∈ R^{10 \times 60},
        b_k ∈  R^10
        x_k ∈  R^60
    The distribution of x_k is controlled by \beta and,
    .. math::
        W_k ~ N(u_k, 1)
        b_k ~ N(u_k, 1)
    where, u_k ~ N(0, \αlpha)

    The number of samples are obtained by lognorm
    """

    # Step 1. get num_samples for each client by lognorm
    num_samples = get_num_samples(num_clients, sigma,
                                  mean_samples, min_samples, max_samples)

    # Step 2. get prior args: mean of w_k, b_k, x_k
    # mean of w_k
    mean_w = np.random.normal(0, 1, (dim, num_classes)) if iid else \
        np.random.normal(0, alpha, num_clients)
    # mean of b_k
    mean_b = np.random.normal(0, 1, num_classes) if iid else \
        np.random.normal(0, alpha, num_clients)
    # mean of x_k
    mean_x = np.zeros((num_clients, dim))
    B = np.random.normal(0, beta, num_clients)
    diagonal = np.zeros(dim)
    for j in range(dim):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)
    for i in range(num_clients):
        mean_x[i] = np.ones(dim) * B[i] if iid else \
            np.random.normal(B[i], 1, dim)

    # Step 3. generate synthetic data
    x_split, y_split = [], []
    for i in range(num_clients):
        # generate w_k, b_k, and x_k with size=num_samples
        w_k = mean_w if iid else \
            np.random.normal(mean_w[i], 1, (dim, num_classes))
        b_k = mean_b if iid else \
            np.random.normal(mean_b[i], 1, num_classes)
        x_k = np.random.multivariate_normal(mean_x[i], cov_x, num_samples[i])
        # calculate y_k by
        y_k = np.zeros(num_samples[i])
        for j in range(num_samples[i]):
            y_k[j] = np.argmax(softmax(np.dot(x_k[j], w_k) + b_k))
        # record generate data convert as lists
        x_split.append(x_k.tolist())
        y_split.append(y_k.tolist())

    # Step 4. divide into train and test and warp as leaf structure
    user_names = [f'client_{i:d}' for i in range(num_clients)]
    train_data = {'users': user_names, 'user_data': {}, 'num_samples': []}
    test_data = {'users': user_names, 'user_data': {}, 'num_samples': []}
    for i, uname in enumerate(user_names):
        idx = int(len(x_split[i]) * train_frac)
        train_data['user_data'][uname] = {'x': x_split[i][:idx], 'y': y_split[i][:idx]}
        test_data['user_data'][uname] = {'x': x_split[i][idx:], 'y': y_split[i][idx:]}
        train_data['num_samples'] = len(train_data['user_data'][uname]['y'])
        test_data['num_samples'] = len(test_data['user_data'][uname]['y'])

    return train_data, test_data


# -- util functions of create datasets


def get_num_samples(num_clients: int, sigma: float, mean_samples: int,
                    min_samples: int, max_samples: int):
    """get num_samples for each client by np.random.lognormal"""
    num_samples = (np.random.lognormal(mean=np.log(mean_samples) + 1e-3,  # add 1e-3 to avoid 0 error of log norm
                                       sigma=sigma,
                                       size=num_clients)).astype(int)
    # make sure num_samples >= min_num_samples,
    num_samples = np.where(num_samples < min_samples, min_samples, num_samples)
    # make sure <= max_num_samples
    num_samples = np.where(num_samples > max_samples, max_samples, num_samples)
    return num_samples


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def polt_data(data: dict, nrows: int = 4, ncols: int = 4):
    """plot json data (at least dim=2) for validation"""
    # random get user_data
    idx = random.randint(0, len(data['users']))
    user_data = data['user_data'][data['users'][idx]]

    # random select samples from user_data
    xs, ys = np.array(user_data['x']), np.array(user_data['y'])
    ids = np.random.randint(0, ys.shape[0], nrows * ncols)
    xs, ys = xs[ids], ys[ids]

    # plot it
    fig = plt.figure()
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.tight_layout()
        plt.imshow(xs[i], cmap='gray', interpolation='none')
        plt.title(f"label: {ys[i]}")
        plt.xticks([])
        plt.yticks([])

    plt.show()
    plt.close(fig)


"""3. functions for format conversion and save data"""


def format_as(data: dict, t_format: str):
    if t_format == 'json': return data  # default structure, don't need to process
    for u in data['users']:
        if t_format == 'npy':
            # json -> np.array
            data['user_data'][u]['x'] = np.array(data['user_data'][u]['x'])
            data['user_data'][u]['y'] = np.array(data['user_data'][u]['y'])
        elif t_format == 'pt':
            # json -> torch.Tensor
            data['user_data'][u]['x'] = torch.Tensor(data['user_data'][u]['x'])
            data['user_data'][u]['y'] = torch.Tensor(data['user_data'][u]['y'])
        else:
            raise NotImplementedError(f"Format '{t_format}' not implemented.")
    return data


def get_f_name(dataset_name: str, num_clients: int, num_classes: int, alpha: float, sigma: float,
               mean_samples: int, min_samples: int, max_samples: int, train_frac: float, split: str,
               iid: bool, shard_size: int, dim: int, **kwargs):
    """get file name according to process args"""
    if iid:
        return f"{dataset_name}_iid_clt={num_clients}_s={sigma}" \
               f"_mu={mean_samples}_r=[{min_samples},{max_samples}]_tf={train_frac}"
    if split == 'leaf':
        return f"{dataset_name}_{split}_clt={num_clients}_r=[{min_samples},{max_samples}]"
    elif split == 'dirichlet':
        f_name = f"{dataset_name}_{split}_clt={num_clients}_cls={num_classes}" \
                 f"_a={alpha}_s={sigma}_mu={mean_samples}_r=[{min_samples},{max_samples}]" \
                 f"_tf={train_frac}"
        if dataset_name == 'synthetic':
            f_name = f_name + f'_dim={dim}'
        return f_name
    elif split == 'shard':
        return f"{dataset_name}_{split}_clt={num_clients}_cls={num_classes}" \
               f"_sz={shard_size}_s={sigma}_mu={mean_samples}_r=[{min_samples},{max_samples}]" \
               f"_tf={train_frac}"


"""4. functions for read data"""


def read_leaf_by_conf(conf_path):
    """read leaf data by yaml conf path

    Return:
        clients: list of client names
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    save_dir, f_name, _ = get_f_by_conf(conf_path)
    train_clients, train_data, test_data = read_leaf_data(save_dir, f_name)
    return train_clients, train_data, test_data


def read_as_torch_dataset(conf_path, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    """read leaf data by conf_path and convert it as torch.util.data.Dataset

    Args:
        conf_path: path of config file, e.g. './template_conf/dirichlet.yaml'
        transform: function/transform for data
        target_transform: function/transform for targets

    Return:
        clients: list of client names
        train_datasets: clients' train data
        test_datasets: clients' test data
        all_train_dataset: sum all clients' train data
        all_test_dataset: sum all clients' test data
    """
    # read data as list
    clients, clients_train, clients_test = read_leaf_by_conf(conf_path)

    # get all set
    all_train_x, all_train_y = [], []
    all_test_x, all_test_y = [], []
    for c in clients:
        all_train_x += clients_train[c]['x']
        all_train_y += clients_train[c]['y']
        all_test_x += clients_test[c]['x']
        all_test_y += clients_test[c]['y']

    # shuffle list
    # train_ids, test_ids = [i for i in range(len(all_train_y))], [i for i in range(len(all_test_y))]
    # random.shuffle(train_ids)
    # random.shuffle(test_ids)
    # all_train_x, all_test_x = np.array(all_train_x)[train_ids].tolist(), np.array(all_test_x)[test_ids].tolist()
    # all_train_y, all_test_y = np.array(all_train_y)[train_ids].tolist(), np.array(all_test_y)[test_ids].tolist()

    # get transform for torch dataset
    conf = read_data_conf(conf_path)
    if transform is None and target_transform is None:
        transform, target_transform = get_transform(conf.dataset)

    # warp as torch.util.data.Dataset
    train_datasets = [WrappedDataset(clients_train[c]['x'], clients_train[c]['y'],
                                     transform, target_transform) for c in clients]
    test_datasets = [WrappedDataset(clients_test[c]['x'], clients_test[c]['y'],
                                    transform, target_transform) for c in clients]
    all_train_dataset = WrappedDataset(all_train_x, all_train_y, transform, target_transform)
    all_test_dataset = WrappedDataset(all_test_x, all_test_y, transform, target_transform)
    return clients, train_datasets, test_datasets, all_train_dataset, all_test_dataset


# -- util functions and class for read data

def read_data_conf(conf_path):
    """read data conf by path"""
    # load conf
    default_conf = OmegaConf.load(DEFAULT_CONF_PATH)
    conf = OmegaConf.load(conf_path)
    conf = OmegaConf.merge(default_conf, conf)
    # check conf
    conf = check_conf(conf)
    return conf


def get_f_by_conf(conf_path):
    """get save dir, file name, and format by yaml config path"""
    conf = read_data_conf(conf_path)
    f_name = get_f_name(**conf)
    return conf.save_dir, f_name, conf.format


def read_leaf_file(file):
    """read leaf data from file"""
    assert file.endswith('.json'), "only '.json' data can be read."
    data = defaultdict(lambda: None)
    with open(file, 'r') as inf:
        cdata = json.load(inf)
    data.update(cdata['user_data'])
    clients = list(sorted(data.keys()))
    return clients, data


def read_leaf_data(data_dir, f_name):
    """parses train and test of given file name.
    Note the file name can be obtained from 'get_f_name'.

    Return:
        clients: list of client ids
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_path = os.path.join(data_dir, f"train-{f_name}.json")
    test_path = os.path.join(data_dir, f"test-{f_name}.json")

    # print(os.path.abspath(train_path))

    train_clients, train_data = read_leaf_file(train_path)
    test_clients, test_data = read_leaf_file(test_path)

    assert train_clients == test_clients
    if 'reddit' in f_name:
        train_data = process_redit(train_data, train_clients)
        test_data = process_redit(test_data, test_clients)

    return train_clients, train_data, test_data


def process_redit(data, clients):
    for c in clients:
        new_train_x, new_train_y = [], []
        for sd in data[c]['x']:
            for d in sd:
                new_train_x.append(d)
        for sd in data[c]['y']:
            for d in sd['target_tokens']:
                new_train_y.append(d)
        data[c]['x'], data[c]['y'] = new_train_x, new_train_y
    return data


class WrappedDataset(Dataset):
    """Wraps list into a pytorch dataset


    Args:
        data (list): feature data list.
        targets (list): target list.
        transform (callable, optional): A function/transform that takes in a data sample
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, data: list,
                 targets: list = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):

        if targets is not None: assert len(targets) == len(data)

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        feature = self.data[idx]
        target = self.targets[idx] if self.targets is not None \
            else self.targets

        if self.transform is not None:
            feature = self.transform(feature)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feature, target


"""5. functions for create dataset"""


def create_by_path(conf_path: str, verbose: int = 1):
    """create data according to config path

    Args:
        conf_path: path of config file, end with .yaml
        verbose: print log?  0 for nothing; 1 for specific; 2 for samplify.
    """
    # load conf
    default_conf = OmegaConf.load(DEFAULT_CONF_PATH)
    conf = OmegaConf.load(conf_path)
    conf = OmegaConf.merge(default_conf, conf)

    # check conf
    conf = check_conf(conf)

    # check if data created
    _, f_name, _ = get_f_by_conf(conf_path)
    train_path = os.path.join(conf.save_dir, f"train-{f_name}.json")
    test_path = os.path.join(conf.save_dir, f"test-{f_name}.json")
    is_created = os.path.exists(train_path) and os.path.exists(test_path)

    if verbose > 0: print(f"{f_name} created? ---> {is_created}")

    # if not create
    if not is_created:
        if verbose > 0: print(f"creating {f_name}!")
        create(conf, verbose)
        if verbose > 0: print(f"{f_name} created!")


def create_batch(conf_dir: str, verbose: int = 2):
    """create a batch of data according to all conf.yaml in given dir

    Args:
        conf_dir: dir of config files, end with .yaml
        verbose: print log?  0 for nothing; 1 for specific; 2 for samplify.
    """
    files = os.listdir(conf_dir)
    files = [f for f in files if f.endswith('.yaml')]
    for f in files: create_by_path(os.path.join(conf_dir, f), verbose)
