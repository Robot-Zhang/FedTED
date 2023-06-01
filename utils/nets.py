import copy
from abc import ABC, abstractmethod
from typing import List
import math
import torch
import torchvision.models
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from vit_pytorch import ViT
from datasets.transforms.nlp_utils import get_word_emb_arr

__all__ = ['create_model',
           'logistic_reg', 'mlp',
           'cnn', 'letnet', 'alexnet', 'resnet', 'resnet8', 'resnet18', 'resnet152',
           'mobilenet_v2', 'shufflenet_v2', 'squeezenet',
           'lstm', 'gru', 'rnn', 'transformer', 'vit',
           'mnist_lstm', 'mnist_gru', 'mnist_rnn',
           'twin_branch', 'TwinBranchNets',
           'generator', ]


def create_model(conf_path, num_clients, num_classes, heterogeneous:bool=False, verbose=0):
    """create models by config path

    Return:
        clients_models: models list of clients
        server_model: model of server
        other_model_dict: dict of other models
    """
    conf = OmegaConf.load(conf_path)

    if verbose:
        print()
        print("=" * 30 + "{:^20}".format("Model Configs") + "=" * 30)
        print(OmegaConf.to_yaml(conf))
        print("=" * 80)
        print()

    other_model_dict = {}

    # # create server and client models
    # model_conf = conf.clients_model[0]
    # models = [eval(model_conf.name)(**model_conf.args) for _ in range(num_clients + 1)]

    if heterogeneous:
        num_conf_models = sum([client_model_conf.num for client_model_conf in conf.clients_model])
        assert num_conf_models >= num_clients, \
            f"{num_clients} clients need model, but only {num_conf_models} configs given. Check `model_conf.yaml`"

    # create client models
    models = []
    for model_conf in conf.clients_model:
        models += [eval(model_conf.name)(**model_conf.args) for _ in range(model_conf.num)]

    # use -1 client model conf create server model
    model_conf = conf.clients_model[-1]
    server_model = eval(model_conf.name)(**model_conf.args)

    # create other models
    gen_conf = conf.other_model.generator
    other_model_dict['generator'] = eval(gen_conf.name)(n_classes=num_classes, **gen_conf.args)
    return models[:num_clients], server_model, other_model_dict


# --------------------
# Compose Net
# --------------------

class TwinBranchNets(nn.Module):
    """Compose feature extractor and classifier into a twin branch net.
        A twin classifier will be made for classifier.

    Args:
        feature_extractor: feature extractor, e.g. conv
        classifier: take output of f_extractor as input, and divide into classes

    Call:
        forward(self, x, use_twin=False), if use_twin, twin_classifier will be used for output.
    """

    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module):
        super(TwinBranchNets, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        # Auxiliary classifier, take same structure as classifier.
        # E.g., in FedRod, this is the personalized head, while classifier is generic head.
        self.twin_classifier = copy.deepcopy(classifier)
        self.use_twin = False

    def forward(self, x):
        feature = self.feature_extractor(x)
        x = self.classifier(feature)
        if self.use_twin:
            x += self.twin_classifier(feature)
        return x


def twin_branch(feature_extractor: OmegaConf, classifier: OmegaConf):
    fe = eval(feature_extractor.name)(**feature_extractor.args)
    cls = eval(classifier.name)(**classifier.args)
    return _twin_branch(fe, cls)


def _twin_branch(feature_extractor: nn.Module, classifier: nn.Module):
    return TwinBranchNets(feature_extractor=feature_extractor,
                          classifier=classifier)


"""FC(MLP) and LR(logistic regression)"""


class FC(nn.Module):
    def __init__(self, in_features=10, out_dim=10, hidden_layers: [int] = None):
        """
         the layers num is len(hidden_layers)+1
        """
        super(FC, self).__init__()
        if hidden_layers is None:
            hidden_layers = []

        layers = []
        if len(hidden_layers) >= 1:
            in_list = [in_features] + hidden_layers
            out_list = hidden_layers + [out_dim]

            count = 0
            for in_dim, out_dim in zip(in_list, out_list):
                layers += [nn.Linear(in_features=in_dim, out_features=out_dim)]
                if count < len(hidden_layers) - 1:
                    layers += [nn.BatchNorm1d(out_dim)]
                    layers += [nn.Dropout(0.2)]
                    layers += [nn.ReLU()]
                    count += 1
        else:
            layers += [nn.Linear(in_features=in_features, out_features=out_dim, bias=True)]
        self.flatten = nn.Flatten()
        self.fcs = nn.Sequential(*layers)

    def forward(self, x):
        f = self.flatten(x)
        r = self.fcs(f)
        return r


def mlp(in_dim=10, out_dim=10, hidden_layers=None):
    return FC(in_features=in_dim,
              out_dim=out_dim,
              hidden_layers=hidden_layers)


def logistic_reg(in_dim=10, out_dim=10):
    return FC(in_features=in_dim,
              out_dim=out_dim)


"""
The CNN class is according to:
    'Brendan McMahan H, Moore E, Ramage D, Hampson S, Agüera y Arcas B.
    Communication-efficient learning of deep networks from decentralized data.
    Proc 20th Int Conf Artif Intell Stat AISTATS 2017. Published online 2017.
    https://arxiv.org/abs/1602.05629'
    
The Cnn2Layer, Cnn3Layer, Cnn4Layer model is according to:
    'Li D, Wang J. FedMD: Heterogenous Federated Learning via Model Distillation. 
    In: NeurIPS. ; 2019:1-8. http://arxiv.org/abs/1910.03581'
"""


def cnn(in_dim: int = 28, in_channels: int = 1, out_dim: int = 10, channels: List[int] = None):
    if channels is None:
        return CNN(in_dim=in_dim,
                   in_channels=in_channels,
                   out_dim=out_dim)
    elif len(channels) == 2:
        return Cnn2Layer(in_dim=in_dim, out_dim=out_dim, in_channels=in_channels,
                         n1=channels[0], n2=channels[1])
    elif len(channels) == 3:
        return Cnn4Layer(in_dim=in_dim, out_dim=out_dim, in_channels=in_channels,
                         n1=channels[0], n2=channels[1], n3=channels[2])
    elif len(channels) == 4:
        return Cnn4Layer(in_dim=in_dim, out_dim=out_dim, in_channels=in_channels,
                         n1=channels[0], n2=channels[1], n3=channels[2], n4=channels[3])


def _cal_out_dim(w0, kernel_size, padding, stride, pool_kernel_size=None, pool_stride=None, pool_padding=0):
    # cal according to pytorch.nn.Conv2d's doc
    w1 = int((w0 + 2 * padding - kernel_size) / stride + 1)
    # cal  according to pytorch.nn.AvgPool2d's doc
    pool_stride = pool_stride if (pool_stride is not None) else pool_kernel_size
    if pool_kernel_size is not None:
        w1 = int((w1 + 2 * pool_padding - pool_kernel_size) / pool_stride + 1)
    return w1


class CNN(nn.Module):
    """
    According to:
        'Brendan McMahan H, Moore E, Ramage D, Hampson S, Agüera y Arcas B.
        Communication-efficient learning of deep networks from decentralized data.
        Proc 20th Int Conf Artif Intell Stat AISTATS 2017. Published online 2017.
        https://arxiv.org/abs/1602.05629'

    Recommend training params:
        lr = 0.1
        Optim = SGD, According to my test, when Optim = Adam, this model can only reach 60.9% accuracy.

    """

    def __init__(self, in_dim=28, in_channels=1, out_dim=10):
        super(CNN, self).__init__()
        # ============model blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(5, 5),
                      padding=1, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=(2, 2), padding=1),
        )
        conv_out_dim = _cal_out_dim(in_dim, kernel_size=5, stride=1, padding=1,
                                    pool_kernel_size=2, pool_padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5),
                      padding=1, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            nn.Flatten()
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=5, stride=1, padding=1,
                                    pool_kernel_size=2, pool_padding=1)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * conv_out_dim ** 2, out_features=512, bias=False),
            nn.ReLU(True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=out_dim, bias=False),
            # nn.ReLU(True),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Cnn4Layer(nn.Module):
    def __init__(self, in_dim=28, out_dim=10, in_channels=1, n1=64, n2=64, n3=64, n4=64, dropout_rate=0.2):
        super(Cnn4Layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size=3, stride=1, padding=1),  # same padding：padding=(kernel_size-1)/2，
            nn.BatchNorm2d(n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        conv_out_dim = _cal_out_dim(in_dim, kernel_size=3, padding=1, stride=1, pool_kernel_size=2, pool_stride=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(n1, n2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, padding=0, stride=2, pool_kernel_size=2, pool_stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(n2, n3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, padding=0, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(n3, n4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, padding=0, stride=2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n4 * conv_out_dim ** 2,
                      out_features=out_dim, bias=False),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


class Cnn3Layer(nn.Module):
    def __init__(self, in_dim=28, out_dim=10, in_channels=1, n1=128, n2=192, n3=256, dropout_rate=0.2):
        super(Cnn3Layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=(1, 1))
        )
        conv_out_dim = _cal_out_dim(in_dim, kernel_size=3, stride=1, padding=1,
                                    pool_kernel_size=2, pool_stride=1, pool_padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(n1, n2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, stride=2, padding=0,
                                    pool_kernel_size=2, pool_stride=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(n2, n3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, stride=2, padding=0)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n3 * conv_out_dim ** 2, out_features=out_dim, bias=False),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


class Cnn2Layer(nn.Module):
    def __init__(self, in_dim=28, out_dim=10, in_channels=1, n1=128, n2=256, dropout_rate=0.2):
        super(Cnn2Layer, self).__init__()

        # same padding：padding=(kernel_size-1)/2，
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool2d(kernel_size=2, stride=1, padding=(1, 1))
        )
        conv_out_dim = _cal_out_dim(in_dim, kernel_size=3, stride=1, padding=1,
                                    pool_kernel_size=2, pool_stride=1, pool_padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(n1, n2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        conv_out_dim = _cal_out_dim(conv_out_dim, kernel_size=2, stride=2, padding=0)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n2 * conv_out_dim ** 2, out_features=out_dim, bias=False),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


"""
The LeNet mode is modified according to: https://github.com/Fong1992/Pytorch-LetNet
"""


# --------------------
# LeNet
# --------------------
class LeNet(nn.Module):
    def __init__(self, in_dim=32, in_channels=3, out_dim=10):
        super(LeNet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self._create_conv_layers([6, 'M', 16, 'M', 120])

        self.fcs = nn.Sequential(
            nn.Linear(120 * (int(in_dim / 8) ** 2), 84),
            nn.ReLU(),
            nn.Linear(84, out_dim)
        )

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fcs(x)
        return x


def letnet(in_dim: int = 28, in_channels: int = 1, out_dim: int = 10):
    return LeNet(in_dim=in_dim,
                 in_channels=in_channels,
                 out_dim=out_dim)


"""
AlexNet, ResNet18, ResNet152, MobileNet V2, ShuffleNet V2, and SqueezeNet are Modified with reference to 'torchvision.models'
Where torchvision.__version__ = '0.8.2+cu101'
"""


# --------------------
# AlexNet
# --------------------
class AlexNet(nn.Module):
    def __init__(self, in_channels=3, out_dim=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Params of first two layers has been changed, including: kernel_size, stride, and padding.
            # Otherwise, CIFAR、MNIST (size only 32/28) predictions cannot be made
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(in_channels: int = 1, out_dim: int = 10):
    return AlexNet(in_channels=in_channels,
                   out_dim=out_dim)


# --------------------
# ResNet18, ResNet152
# --------------------
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels=3, out_dim=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet8(in_channels=3, out_dim=10, **kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1],
                  in_channels=in_channels,
                  out_dim=out_dim, **kwargs)


def resnet18(in_channels=3, out_dim=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  in_channels=in_channels,
                  out_dim=out_dim, **kwargs)


def resnet152(in_channels=3, out_dim=10, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3],
                  in_channels=in_channels,
                  out_dim=out_dim, **kwargs)


def resnet(in_channels=3, out_dim=10, layers=8, **kwargs):
    if layers == 8:
        return resnet8(in_channels=in_channels, out_dim=out_dim, **kwargs)
    elif layers == 18:
        return resnet8(in_channels=in_channels, out_dim=out_dim, **kwargs)
    elif layers == 152:
        return resnet152(in_channels=in_channels, out_dim=out_dim, **kwargs)
    else:
        raise NotImplementedError


# --------------------
# MobileNet V2
# --------------------
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_dim=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            out_dim (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(in_channels, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, out_dim),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(in_channels=3, out_dim=10, **kwargs):
    return MobileNetV2(in_channels=in_channels,
                       out_dim=out_dim,
                       **kwargs)


# --------------------
# ShuffleNet V2
# --------------------
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleNetInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ShuffleNetInvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, in_channels=3, out_dim=1000,
                 inverted_residual=ShuffleNetInvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, out_dim)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def shufflenet_v2(in_channels=3, out_dim=1000, **kwargs):
    return ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024],
                        in_channels=in_channels,
                        out_dim=out_dim,
                        **kwargs)


# --------------------
# SqueezeNet V2
# --------------------
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', in_channels=3, out_dim=1000):
        super(SqueezeNet, self).__init__()
        self.out_dim = out_dim
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.out_dim, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def squeezenet(in_channels=3, out_dim=10, **kwargs):
    return SqueezeNet(in_channels=in_channels,
                      out_dim=out_dim, **kwargs)


"""
LSTM, GRU, RNN 
"""


def lstm(vocab_size=80, out_dim=80, embedding_dim=8, hidden_size=256, num_layers=2, dropout=0.2,
         bidirectional=False, tie_weights=False, embedding_path=None, seq2seq=False, use_embedding=True):
    embedding_weights = _get_embedding_weights(embedding_path)

    return RNNModel('LSTM', vocab_size, out_dim, embedding_dim, hidden_size, num_layers, dropout,
                    bidirectional, tie_weights, embedding_weights, seq2seq=seq2seq,
                    use_embedding=use_embedding)


def gru(vocab_size=80, out_dim=80, embedding_dim=8, hidden_size=256, num_layers=2, dropout=0.2,
        bidirectional=False, tie_weights=False, embedding_path=None, seq2seq=False, use_embedding=True):
    embedding_weights = _get_embedding_weights(embedding_path)

    return RNNModel('GRU', vocab_size, out_dim, embedding_dim, hidden_size, num_layers, dropout,
                    bidirectional, tie_weights, embedding_weights, seq2seq=seq2seq,
                    use_embedding=use_embedding)


def rnn(vocab_size=80, out_dim=80, embedding_dim=8, hidden_size=256, num_layers=2, dropout=0.2, bidirectional=False,
        tie_weights=False, embedding_path=None, nonlinearity='relu', seq2seq=False, use_embedding=True):
    assert nonlinearity in ['relu', 'tanh'], f"nonlinearity {nonlinearity} error"

    embedding_weights = _get_embedding_weights(embedding_path)

    return RNNModel('GRU', vocab_size, out_dim, embedding_dim, hidden_size, num_layers, dropout,
                    bidirectional, tie_weights, embedding_weights, nonlinearity, seq2seq=seq2seq,
                    use_embedding=use_embedding)


def _get_embedding_weights(embedding_path):
    if embedding_path is None:
        embedding_weights = None
    else:
        embedding_weights, _, _ = get_word_emb_arr(embedding_path)
        embedding_weights = torch.from_numpy(embedding_weights)
    return embedding_weights


class RNNModel(nn.Module):
    """
    Container module with an encoder (embedding), a recurrent module (LSTM, GRU, RNN), and a decoder (FC).


    """

    def __init__(self, rnn_type, vocab_size, output_dim, embedding_dim, hidden_size, num_layers, dropout=0.2,
                 bidirectional=False,
                 tie_weights=False, embedding_weights=None, nonlinearity='relu', seq2seq=False,
                 use_embedding=True):

        super(RNNModel, self).__init__()

        # encoder
        self.use_embedding = use_embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)

        # RNN
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_size, num_layers,
                                             dropout=dropout, bidirectional=bidirectional,
                                             batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, nonlinearity=nonlinearity,
                              dropout=dropout, bidirectional=bidirectional, batch_first=True)
        else:
            raise NotImplementedError

        if bidirectional:
            hidden_size *= 2

        # decoder
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )

        # softmax
        # self.softmax = nn.Softmax(dim=-1)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            assert hidden_size == embedding_dim, \
                'When using the tied flag, hidden_size must be equal to embedding_dim'
            self.fc.weight = self.embedding.weight

        # init weights
        if embedding_weights is not None:
            self.embedding.from_pretrained(embedding_weights)
        # init_range = 0.1
        # if embedding_weights is None:
        #     nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        # else:
        #     self.embedding.from_pretrained(embedding_weights)
        # nn.init.zeros_(self.fc.bias)
        # nn.init.uniform_(self.fc.weight, -init_range, init_range)

        self.seq2seq = seq2seq
        self.out_dim = output_dim
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        # batch first, i.e. [B, S, D]
        emb = self.dropout(self.embedding(x)) if self.use_embedding else x

        if hidden is None:
            output, _ = self.rnn(emb)
        else:
            output, hidden = self.rnn(emb, hidden)

        if not self.seq2seq:  # final hidden state for output
            output = output[:, -1, :]

        decoded = self.fc(output)

        if self.seq2seq:
            batch_size = x.size()[0]
            decoded = decoded.view(batch_size, -1, self.out_dim)

        if hidden is None:
            return decoded  # F.log_softmax(decoded, dim=-1)  # self.softmax(decoded)
        else:
            return decoded, hidden  # F.log_softmax(decoded, dim=-1), hidden  # self.softmax(decoded), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_size)


"""Vision-RNN"""


class Vision_RNN(nn.Module):
    def __init__(self, rnn_type='LSTM', channels=1, in_dim=28, n_classes=10, hidden_size=64, num_layers=1,
                 bidirectional=False, nonlinearity='relu'):
        super(Vision_RNN, self).__init__()
        self.channels = channels

        # RNN
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size=in_dim, hidden_size=hidden_size,
                                             num_layers=num_layers,
                                             bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers,
                              nonlinearity=nonlinearity,
                              bidirectional=bidirectional, batch_first=True)
        else:
            raise NotImplementedError

        if bidirectional:
            hidden_size = hidden_size * 2

        self.batch_norm = nn.BatchNorm1d(hidden_size * self.channels)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(hidden_size * self.channels, int(hidden_size / 2) * self.channels)
        self.fc2 = nn.Linear(int(hidden_size / 2) * self.channels, n_classes)

    def forward(self, input):
        # Shape of input is (batch_size, channels, in_dim, in_dim)
        outputs = []
        for channel in range(self.channels):
            # for each channel, take input as (batch_size, in_dim, in_dim)
            # as required by RNN when batch_first is set True
            x = input[:, channel, :, :]
            output, _ = self.rnn(x)
            # RNN output shape is (seq_len, batch, input_size)
            # Get last output of RNN
            output = output[:, -1, :]
            outputs.append(output)
        output = torch.cat(outputs, dim=1)

        # put together
        output = self.batch_norm(output)
        output = self.dropout1(output)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.dropout2(output)
        output = self.fc2(output)
        # output = F.log_softmax(output, dim=1)
        return output


# Note: vi_rnn like is good in mnist-like data, but not good in cifars or harder images.

def mnist_lstm(in_dim=28, channels=1, out_dim=10, hidden_size=64, num_layers=1,
               bidirectional=False):
    return Vision_RNN(rnn_type='LSTM',
                      channels=channels,
                      in_dim=in_dim,
                      n_classes=out_dim,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      bidirectional=bidirectional
                      )


def mnist_gru(in_dim=28, channels=1, out_dim=10, hidden_size=64, num_layers=1,
              bidirectional=False):
    return Vision_RNN(rnn_type='GRU',
                      channels=channels,
                      in_dim=in_dim,
                      n_classes=out_dim,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      bidirectional=bidirectional
                      )


def mnist_rnn(in_dim=28, channels=1, out_dim=10, hidden_size=64, num_layers=1,
              bidirectional=False, nonlinearity='relu'):
    return Vision_RNN(rnn_type='RNN',
                      channels=channels,
                      in_dim=in_dim,
                      n_classes=out_dim,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      bidirectional=bidirectional,
                      nonlinearity=nonlinearity
                      )


"""Transformer"""

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        # >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    # ntoken->vocab_size, ninp->embedding_dim, nhid->hidden_size, ntoken->output_dim
    def __init__(self, vocab_size, output_dim, embedding_dim, num_heads, dropout, hidden_size,
                 num_layers, seq2seq=False, use_embedding=True, embedding_weights=None):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
                              'lower.') from e
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, output_dim)
        )

        # self.softmax = nn.Softmax(dim=-1)

        # self.init_weights()
        self.use_embedding = use_embedding
        self.seq2seq = seq2seq
        if embedding_weights is not None:
            self.encoder.from_pretrained(embedding_weights)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):

        seq_len = src.size(1)
        src = src.view(seq_len, -1)
        # input src is batch_first, i.e. [batch_size, seq_len]
        # seq_len = src.size(1)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = _generate_square_subsequent_mask(len(src)).to(device)
                # if self.src_mask is None or self.src_mask.size(0) != seq_len:  # we are batch first
                #     mask = _generate_square_subsequent_mask(seq_len).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        emb = self.encoder(src) if self.use_embedding else src
        src = emb * math.sqrt(self.embedding_dim)

        # view as seq first
        # src = src.view(seq_len, -1, self.embedding_dim)

        # cal with [seq_len, batch_size, embedding_dim]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        if not self.seq2seq:  # final seq output
            output = output[-1, :, :]
            # output = output[:, -1, :]

        output = self.decoder(output)

        if self.seq2seq:
            batch_size = src.size()[0]
            output = output.view(batch_size, -1, self.out_dim)

        return output  # F.log_softmax(output, dim=-1)  # self.softmax(output)


def transformer(vocab_size=80, out_dim=80, embedding_dim=8, num_heads=2, dropout=0.2,
                hidden_size=256, num_layers=2, seq2seq=False, use_embedding=True, embedding_path=None):
    embedding_weights = _get_embedding_weights(embedding_path)
    return TransformerModel(vocab_size, out_dim, embedding_dim, num_heads, dropout,
                            hidden_size, num_layers, seq2seq, use_embedding, embedding_weights)


"""ViT"""


def vit(in_dim: int = 28, channels: int = 3, patch_size: int = 4, embedding_dim: int = 1024,
        num_layers=2, num_heads=16, hidden_size: int = 1024, dropout=0.1, emb_dropout: int = None,
        out_dim: int = 10):
    if emb_dropout is None: emb_dropout = dropout

    return ViT(
        image_size=in_dim,
        channels=channels,
        patch_size=patch_size,
        num_classes=out_dim,
        dim=embedding_dim,
        depth=num_layers,
        heads=num_heads,
        mlp_dim=hidden_size,
        dropout=dropout,
        emb_dropout=emb_dropout
    )




"""
Generators
"""


def generator(n_classes, noise_dim, hidden_dim, latent_dim, embedding=False):
    return Generator(n_classes, noise_dim, hidden_dim, latent_dim, embedding)


class Generator(nn.Module):
    def __init__(self, n_classes, noise_dim, hidden_dim, latent_dim, embedding=False):
        """Generator according to FedGen Project: https://github.com/zhuangdizhu/FedGen"""

        super(Generator, self).__init__()

        self.hidden_dim, self.latent_dim, self.n_classes, self.noise_dim, self.embedding = \
            hidden_dim, latent_dim, n_classes, noise_dim, embedding

        # input dim
        input_dim = self.noise_dim + self.n_classes
        if self.embedding:
            # for nlp task, vector data as self.noise_dim
            self.embedding_layer = nn.Embedding(self.n_classes, self.noise_dim)
            input_dim = self.noise_dim * 2

        # fc layers with bn
        fc_configs = [input_dim, self.hidden_dim]
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_configs) - 1):
            input_dim, out_dim = fc_configs[i], fc_configs[i + 1]
            # print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]

        # output latent
        self.representation_layer = nn.Linear(fc_configs[-1], self.latent_dim)

    def get_number_of_parameters(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def forward(self, labels):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :return: a dictionary of output information.
        """
        batch_size = labels.size(0)
        # sampling from Gaussian
        eps = torch.randn((batch_size, self.noise_dim)).to(labels.device)

        # embedded dense vector
        if self.embedding:
            y_input = self.embedding_layer(labels)
        else:  # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_classes).to(labels.device)
            y_input.zero_()
            # labels_int64 = labels.type(torch.LongTensor)
            y_input.scatter_(1, labels.view(-1, 1), 1)
            # y_input = labels.view_as(torch.FloatTensor(batch_size, self.n_classes))
        z = torch.cat((eps, y_input), dim=1)

        for layer in self.fc_layers:
            z = layer(z)

        z = self.representation_layer(z)
        return z, eps
