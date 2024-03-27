"""Base train functions for reusing
"""
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


def distill_by_models(student_model: nn.Module, teacher_models: [nn.Module], dataloader, optimizer, loss_fn,
                      epochs=10, device='cpu', weights: List = None, verbose=0):
    """
    https://github.com/AberHu/Knowledge-Distillation-Zoo
    https://github.com/haitongli/knowledge-distillation-pytorch
    """
    # step 1. init models and teacher weights
    student_model.to(device)
    student_model.train()
    for model in teacher_models:
        model.to(device)
        model.eval()

    weights = [1.0 / len(teacher_models) for _ in range(len(teacher_models))] \
        if weights is None else weights

    # step 2. train loop
    loss_metric = []  # to record avg loss
    for epoch in range(epochs):
        # init loss value
        loss_value, num_samples = 0, 0
        # one epoch train
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            student_out = student_model(x)
            with torch.no_grad():
                teacher_out = sum(
                    [teacher_models[idx](x) * weights[idx]
                     for idx in range(len(teacher_models))]
                )
            loss = loss_fn(student_out, teacher_out)
            # backward & step optim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # get loss valur of current bath
            loss_value += loss.item()
            num_samples += _.size(0)
        loss_value = loss_value / num_samples
        loss_metric.append(loss_value)
        if verbose == 1:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_value:.4f}')

    # step 3. release gpu resource
    for model in teacher_models + [student_model]:
        model.to('cpu')
    student_model.eval()
    torch.cuda.empty_cache()

    avg_loss = sum(loss_metric) / len(loss_metric)
    return avg_loss


def train_model(model: nn.Module, train_loader: DataLoader,
                optimizer=None, loss_fn=nn.CrossEntropyLoss(),
                epochs=10, device='cpu', verbose=0, one_batch=False,
                test_loader=None, test_interval=10):
    """

    """
    # rationalization proposal
    assert epochs > 0
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # step 1. model init
    model.to(device)
    model.train()
    # step 2. train loop
    loss_metric = []  # to record avg loss
    for epoch in range(epochs):
        # init loss value
        loss_value, num_samples = 0, 0
        # one epoch train
        for i, (x, y) in enumerate(train_loader):
            # put tensor into same device
            x, y = x.to(device), y.to(device)
            # calculate loss
            y_ = model(x)
            loss = loss_fn(y_, y)
            # backward & step optim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # get loss valur of current bath
            loss_value += loss.item()
            num_samples += y.size(0)
            if one_batch:
                break

            # ============   Dissussion   ============
            # In fact, we think in FL settings, all data should be used, this it the meaning of it,
            # and consist with original FedAvg. Therefore, this part is dropped.
            # if not use_all:
            #     # In some FL projects like FedGen, the local train is only a small part.
            #     # In vanilla FedAvg, each local epoch need to train through hole local dataset.
            #     break

        # Use mean loss value of each epoch as metric
        # Just a reference value, not precise. If you want precise, dataloader should set `drop_last = True`.
        loss_value = loss_value / num_samples
        # if verbose, print training metrics.
        if verbose == 1:
            if test_loader is not None and (epoch + 1) % test_interval == 0:
                accuracy, loss_value = evaluate_model(model, test_loader, loss_fn, device=device, release=False)
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_value:.4f}, Accuracy: {accuracy:.3f}')
                model.to(device)
                model.train()
            else:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_value:.4f}')
        loss_metric.append(loss_value)
    # step 3. release gpu resource
    model.to('cpu')
    torch.cuda.empty_cache()

    avg_loss = sum(loss_metric) / len(loss_metric)
    return avg_loss


def evaluate_model(model: nn.Module, test_loader, loss_fn=nn.CrossEntropyLoss(),
                   metric_type='accuracy', device='cpu', verbose=0, release=True):
    """
    """

    # rationalization proposal
    assert metric_type in ['accuracy', 'mse']

    # init model with eval mode
    model.eval()
    model.to(device)

    # init metric and loss value
    loss_value, accuracy, num_samples = 0, 0, 0

    # test by test loader
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # forward
            y_ = model(x)
            # record loss value
            loss_value += loss_fn(y_, y).item()
            # metric correct
            if metric_type == 'accuracy':
                predicted = y_.argmax(dim=1, keepdim=True)
                accuracy += predicted.eq(y.view_as(predicted)).sum().item()
            elif metric_type == 'mse':
                accuracy += F.mse_loss(y_, y, reduction='sum').item()
            num_samples += y.size(0)

    # cal metric
    loss_value = loss_value / len(test_loader)
    accuracy = accuracy / num_samples

    # release gpu resource
    if release:
        model.to('cpu')
        torch.cuda.empty_cache()

    if verbose == 1:
        print(f'Test by {num_samples:d} samples, loss: {loss_value:.4f}, {metric_type}:{accuracy:.3f}')
    return accuracy, loss_value


def freeze(model, freeze_name=None):
    """unfreeze model parameters"""
    set_requires_grad(model, freeze_name, False)


def unfreeze(model, unfreeze_name=None):
    """unfreeze model parameters"""
    set_requires_grad(model, unfreeze_name, True)


def set_requires_grad(model, param_name=None, requires_grad=False):
    for name, param in model.named_parameters():
        if param_name is None:
            param.requires_grad = requires_grad
        else:
            if param_name in name:
                param.requires_grad = requires_grad

