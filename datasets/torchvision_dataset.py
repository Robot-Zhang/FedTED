from torchvision import datasets, transforms

"""
Load raw dataset of torchvision
"""
DATA_PATH = "./datasets/raw/"


def get_dataset_by_name(name, is_train):
    return eval(name)(is_train)


def MNIST(is_train: bool):
    return datasets.MNIST(DATA_PATH + "mnist",
                          train=is_train,
                          download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor()
                          ])
                          )


def EMNIST(is_train: bool, split='balanced'):
    return datasets.EMNIST(DATA_PATH + "emnist",
                           split=split,
                           # split='byclass',  # 62 classes
                           # split='bymerge',  # 47 classes
                           # split='balanced',  # 47 classes
                           # split='letters',  # 27 classes
                           # split='digits',  # 10 classes
                           # split='mnist',  # 10 classes
                           train=is_train,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])
                           )

def FashionMNIST(is_train: bool):
    return datasets.FashionMNIST(DATA_PATH + "FashionMNIST",
                                 train=is_train,
                                 download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ])
                                 )


def CIFAR10(is_train: bool):
    return datasets.CIFAR10(DATA_PATH + 'cifar10', train=is_train, download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ])
                            )


def CIFAR100(is_train: bool):
    return datasets.CIFAR100(DATA_PATH + 'cifar100', train=is_train, download=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ])
                             )
