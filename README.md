# FedTED

Source code of "Improving Generalization and Personalization in Model-Heterogeneous Federated Learning"

![latent-class](/docs/latent-class.gif)

<!-- TODO: add paper link and author link after pub -->

## Declaration

This is a paper being reviewed, and questions and reuse are welcome, but please indicate the quotation after we publish it.

## 1. Requirements

A suitable [conda](https://conda.io/) environment named `FedTED` can be created
and activated with:

```bash
$ conda create -n FedTED python=3
$ conda activate FedTED
$ conda install pytorch torchvision -c pytorch
$ pip3 install -r requirements.txt
```

## 2. Getting Started

After completing the configuration, you can run as follows.

```bash
$ python main.py --algorithm <alg_name> --exp_conf <exp_conf.yaml> --data_conf <data_conf.yaml> --model_conf <model_conf.yaml> --seed <seed> --device <seed>
```

For example, run FedTED on MNIST with model-heterogeneous settings:

```bash
$ python main.py --algorithm FedTED --exp_conf ./configs/template/exp/het-exp.yaml --data_conf ./configs/template/data/mnist.yaml --model_conf ./configs/template/model/het-mnist.yaml --seed 15698 --device cuda:0
```

## 3. Usage

### 3.1 Arguments

In this project, main.py takes the following arguments:

+ `--algorithm`: name of the implemented algorithms.
+ `--num_clients`: number clients, if not specific, use the value in `data_conf.yaml`
+ `--exp_name`: exp name, sub dir for save log, if not specific, use the value in `exp_conf.yaml`
+ `--exp_conf`: experiment config yaml files
+ `--data_conf`: dataset config yaml files
+ `--public_conf`: public dataset config yaml files, default is None. For FedMD and Kt-pFL. 
+ `--model_conf`: model config yaml files
+ `--device`:  run device (cpu | cuda:x, x:int > 0)
+ `--seed`: random seed
+ `--save_model`: bool, if save model at each test interval.

### 3.2 EXP Config YAML

This is a typic yaml file:

```yaml
# All algorithm in the same experiment use same confing file
# 1. Settings
exp_name: "het-template" # name of experiment
heterogeneous: True
# 2. Basic args for FedAvg
rounds: 100 # communication rounds
epochs: 1 # epochs of local update
loss: 'CrossEntropyLoss' # loss fn name in torch.nn.*
opt: 'Adam'  # optimizer name in torch.optim.*, e.g. Adam, SGD
optim_kwargs: # args for optimizer
  lr: 1e-3 # learning rate of local update
batch_size: 32 # batch_size of local update
sample_frac: 0.5 # select fraction of clients
test_interval: 1
# 3. Optional args for FL algorithms
# ----3.1 Args for Center
center_update_samples: # if not None, means the used samples in each update epoch, recommend as None
# ---- 3.2 Args for FedMD and Kt_pFL
pretrain_epochs: 1   # pretrain epochs in public dataset
num_alignment: 200   # number of alignment data in FedMD/Kt_pFL
distill_lr: 1e-5  # lr for distillation
distill_epochs: 1 # epochs for distillation
distill_temperature: 20 # temperature for distillation
# ---- 3.3 Args for FedDF
#       note: public_data and distill_temperature is same as 3.2
ensemble_epoch: 5
ensemble_lr: 1e-4 # lr for ensemble, suggest lower than lr
# ---- 3.4 Args for FedDistill
fed_distill_gamma: 1e-4 #1e-4 # According to our test, it's value should not be larger than 1e-4
early_exit: 5  # exit the algorithm (FedDistill), and use norm FedAvg.
fed_distill_aggregate: True # if aggregate model weights by avg. If False, vanilla FedDistill (weak but save communication resource), else, FedDistill + FedAvg.
# ---- 3.5 Args for FedGen
#       note: distill_temperature s same as 3.2
generative_alpha: 10.0  # hyper-parameters for clients' local update
generative_beta: 1.0  # hyper-parameters for clients' local update
gen_epochs: 10  # epochs for updating generator
gen_lr: 1e-3  # lr for updating generator
# ---- 3.6 Args for FedFTG,
#       note: gen_epochs, gen_lr is same as 3.5
#             ensemble_epoch, ensemble_lr is same as 3.3
#             distill_temperature is same as 3.2
finetune_epochs: 1
lambda_cls: 1. # hyper-parameters of updating generator
lambda_dis: 1. # hyper-parameters of updating generator
```

### 3.3 Supported Federated Datasets

#### Four types of federated data production are supported

1. Subset of entire Leaf：[Leaf](https://arxiv.org/abs/1812.01097)
2. dirichlet split：[FedProx](https://arxiv.org/abs/1812.06127)
3. create by shard：[FedAvg](https://arxiv.org/abs/1602.05629)
4. synthetic：[FedProx](https://arxiv.org/abs/1812.06127)

#### Partition methods for each dataset

|   Dataset    | leaf | dirichlet | shard |
| :----------: | :--: | :-------: | :---: |
|    MNIST     |      |     Y     |   Y   |
| FashionMNIST |      |     Y     |   Y   |
|    EMNIST    |      |     Y     |   Y   |
|   CIFAR10    |      |     Y     |   Y   |
|   CIFAR100   |      |     Y     |   Y   |
|   femnist    |  Y   |           |       |
|    celeba    |  Y   |           |       |
|    reddit    |  Y   |           |       |
|   sent140    |  Y   |           |       |
|  shakespare  |  Y   |           |       |
|  synthetic   |  -   |     -     |   -   |

### 3.4 Supported Client Models

+ Logistic regression

+ MLP
+ Tested CNN by [FedAvg](https://arxiv.org/abs/1602.05629) and  [FedMD](http://arxiv.org/abs/1910.03581)
+ LetNet
+ AlexNet
+ ResNet
+ MobileNet_v2
+ ShuffleNet_v2
+ SqueezeNet
+ LSTM
+ GRU
+ RNN
+ Transformer
+ ViT
+ Mnist_LSTM
+ Mnist_GRU
+ Mnist_RNN
+ Twin-Branch
+ Generator


## 4. Introduction to FedTED 

FedTED is a model-heterogeneous generalization-personalization balanced framework via **t**win-branch n**e**twork and feature **d**istillation.

### 4.1 Scenario

In model-homogeneous federated learning, expensive communication overhead hinders the deployment of large-scale models, while users like medical institutions require strong models to guarantee high accuracy. In addition, as a valuable asset, users may not willing to upload their models. Especially, directly sharing homogeneous models is vulnerable to backdoor attacks and model poisoning. Moreover, for heterogeneous devices with different capacities, homogeneous models become powerless to adapt to their hardware conditions. Therefore, in addition to data heterogeneity, model heterogeneity should also be considered.

In this FedTED , we try to solve a more challenging problem than before: ***How can Federated Learning enhance both generalization and personalization when clients' models are heterogeneous?*** Overcoming this obstacle, FL will be able to enjoy multiple benefits in meeting personalized needs, integrating generic model, and protecting user privacy.

![background](/docs/background.png)

In the scenario of FedTED, *both clients' data and models are heterogeneous*, which makes it challenging. Due to privacy concerns, users no longer share their local model but upload latent knowledge. Using this knowledge, the server can reconstruct a generic model, while clients can train better-personalized models.

### 4.2 Workflow

Overview of FedTED. It is orchestrated by three novel components: twin-branch local update with knowledge distillation, feature-level heterogeneous aggregation, and generator-based model reconstruction. In the figure, solid arrows represent the forward flow, and dashed arrows represent the backward flow. We use $G(\cdot)$ to denote the feature generator, $E(\cdot)$ to denote the feature extractors, $C(\cdot)$ to denote the predictors, and ${{\hat \sigma }_k}$ to denote the client-specific weight vectors.


Next, we will elaborate on the three novel components of FedTED.
+ **Twin-branch local update with knowledge distillation** is processed in "Clients" part of the figure. It consists of a sampler, a feature extractor ${E_k}({x_k};{w_{ek}})$, and two predictors ${C_g}(z;{w_{pk}})$, ${C_p}(z;{{\tilde w}_{pk}})$. Among them, the sampler selects the appropriate proxy data $\tilde z_k$ based on the local data distribution $p({y_k})$. The selected proxy data $\tilde z_k$ is used as global knowledge to assist in updating the local feature extractor. The predictors ${C_g}(z;{w_{pk}})$ and ${C_p}(z;{{\tilde w}_{pk}})$ are twin task-specific layers that share the same structure. Among these two predictors, ${C_g}(z;{w_{pk}})$ is regarded as generic branch and ${C_p}(z;{{\tilde w}_{pk}})$ is regarded as personalized branch. For personalized tasks, the output of the personalized predictor is directly taken as the predicted value. During the training phase of general tasks, the generic output is corrected by a prior vector ${\hat \sigma }_k$ (determined by the batch data distribution). It's worth noting that only generic predictors are uploaded during cooperative training.
+ **Feature-level heterogeneous aggregation** is processed in "Server" part of the figure. It consists of a feature generator $G(\tilde y, \varepsilon; w_g)$, a server-side generic predictor $C(z,w_p)$, and the client-uploaded predictors $\{C_g(z,w_pk)\}$. The generator takes legal labels $\tilde y$ and random noise $\varepsilon$ as inputs to generate latent features $\tilde z$. The generated latent features and corresponding labels constitute the proxy data. The goal of $G(\tilde y, \varepsilon; w_g)$ is to make the generated features indistinguishable from the real features. The server-side generic predictor $C(z,w_p)$ is averaged by the other predictors $\{C_g(z,w_pk)\}$ and fine-tuned with proxy data. Since most models use a fully connected network as the output layer, the predictors of heterogeneous models can adopt a similar structure. In FedTED, the predictors are assumed to take the same input and output space. This can be applied to arbitrary model-heterogeneous scenarios by simply adding fully connected layers as heads. For example, for heterogeneous models of $10$ classes, a fully connected network with 10 inputs and 10 outputs can be added as the predictor after their output layer to match FedTED.
+ **Generator-based model reconstruction** is processed in "Reconstruction" part of the figure. It consists of a sampler, a feature extractor $E(x;w_e)$, and a generic predictor $C(z,w_p)$. The generic predictor comes from the "Server" component, and the sampler works the same as in "Clients" component. The feature extractor $E(x;w_e)$ is trained by taking the generated features as labels. Here, the generated features can be regarded as the soft labels in the dataset distillation. For the generic clients, the input to the feature extractor comes from their local dataset. For the server, the input comes from public samples, which does not need to be large. After completing the training of the feature extractor, a generic model can be obtained by stacking the feature extractor and the predictor. This generic model can be used to predict new data or be distributed directly to cold start clients.


![framework](/docs/framework.png)

When FedTED is working, the clients first distill their feature extractor under the guidance of the proxy data (flow (1) in the figure). Then, they update the distilled feature extractor and the twin predictors respectively (flow (2) in the figure). In the next step, the server trains a feature generator through the uploaded predictors (flow (3) in the figure). Additionally, a global generic predictor can be obtained by aggregating the predictors uploaded by the clients (flow (4) in the figure). Finally, the generated proxy data can be used to train a feature extractor, which in turn forms a new generic model with the aggregated predictor (flow (5) in the figure). Note that the generic predictor is updated and shared among all clients in the next round of FL. The workflow of FedTED is similar to classical Federated Learning. That is, the steps of client selection, information distribution, local update, parameters upload, and aggregation are performed in sequence. The compatibility with other frameworks is further enhanced by the fact that FedTED only requires generic predictors to be uploaded during cooperative training. This makes it easy to integrate FedTED with any framework that supports Federated Learning.



#### 4.3 Experimental Setup

#### Benchmarks

FedTED has been compared with SOTA algorithms for both model-homogeneous and model-heterogeneous Federated Learning. In addition to the most basic Local (training isolated on clients), Center (centralizing all data to the server), and FedAvg (vanilla Federated Learning), two data-heterogeneous algorithms, FedProx and SCAFFOLD, have been compared. Furthermore, the ensemble distill based FedDF, the SOTA generic-personalized balanced FedRoD, and two data-free distillation algorithms FedFTG and FedGen  are also take into account. Additionally, three model-heterogeneous Federated Learning algorithms, FedDistill, FedMD, and Kt-pFL are involved in the comparison. Among them, FedDistill aggregates label-specific logits, FedMD averages clients' logits of public data, while Kt-pFL weight sum clients' logits by a knowledge transfer coefficient.

#### Datasets and Models
FedTED is evaluated on four different tasks, including image classification, sentiment analysis, next character prediction, and synthetic data classification. The settings of used datasets are shown as follows.

![dataset](/docs/dataset.png)

Specifically, image classification is tested on FEMNIST, CELEBA, MNIST, Fashion-MNIST, and CIFAR10, and the heterogeneous models are configured in the same way as FedMD. Sent140 is used for sentiment analysis, which is extracted from Twitter. Its task is to distinguish whether the text is positive or negative. 
The heterogeneous model adopts 1 to 3 layers of LSTM, GRU and RNN, plus Glove6B as a fixed embedding layer. The next character prediction uses Shakespeare, whose task is to predict what each role in Shakespeare's works said. Here, each role is considered a client. Its model configuration is similar to Sent140, but the embedding layer is trained from scratch. The dataset used for synthetic data classification is the same as that of FedProx, and its heterogeneous model is a multi-layer perceptron with different parameters. For the above datasets, FEMNIST, CELEBA, Sent140, and Shakespeare are processed in the way of LEAF. MNIST, Fashion-MNIST, CIFAR10, and Synthetic control the class equilibrium with the Dirichlet distribution ($\alpha=0.1$) and the clients' sample size with the Log-Normal distribution ($\sigma=0.5$). The structure of these models is shown as follows.

![models](/docs/models.png)

#### Implementation

All of the experiments are conducted using the PyTorch framework. The benchmark algorithms and Federated Learning frameworks are refactored according to their corresponding papers and open-source codes. Our code runs on a Ubuntu 22.04 server with an Intel(R) Core(TM) i9-9900KF CPU@3.6GHz and 4 NVIDIA Tesla V100 GPU cards.

### 4.4 Evaluation

#### Weight Distribution of  Twin Branches

We count the weights of twin branches and get the weight distribution. After training, under the control of multitask regularization terms, the parameter weights of personalized branch and generic branch are calibrated with each other. At the same time, due to the differences in tasks between personalized branch and generic branch, the emphasis of their weights will be different, which enables FedTED to use their outputs as needed and realize synchronous promotion on personalized tasks and generalized tasks.

+ weights of generic branch

![latent-class](/docs/g-weight.gif)

+ weights of personalized branch

![latent-class](/docs/p-weight.gif)

#### Learned Latent Knowledge of FedTED

We take the argmax of latent for each sample as its feature. Among them, the latent features of different classes of data have significant differences. This indicates that when the model converges, the model trained by FedTED can preserve the knowledge of the sample data in a richer form in by latent, which can then be transferred between the client and the server.

![latent-class](/docs/latent-class.gif)

#### Effectiveness of Re-wight

In FedTED, we decouple the twin-branch network of FedTED into two related tasks by a prior conditional corrector. Using this concise re-weight variable, the training effect of FedTED is significantly stabilized. 

 ![re-weight-performance](/docs/re-weight-performance.png)

For more detailed evaluation, , please refer to our paper.

## 5. Contribution Navigation

Too add new FL algorithms, just inherit `FedAvg` class directly, and then modify the corresponding function in the protocol. Include:

```python
"""Protocol of algorithm, in most kind of FL, this is same."""

def sample_clients(self):
    pass

def distribute_model(self):
    pass

def local_update(self, epochs):
    pass

def aggregate(self):
    pass

def test_performance(self, r, is_final_r=False, verbose=0, save_ckpt=False):
    pass
```

For more detailed information, see`./trainer/FedAvg`.



## 6. BibTeX

It will be given after publication.

<!-- TODO: refresh bib after publication-->