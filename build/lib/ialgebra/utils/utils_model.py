#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn

from ialgebra.models import models
from ialgebra.utils.utils_data import load_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0


name2class = {
    'lenet': 'LeNet',
    'resnet18': 'ResNet',
    'resnet50': 'ResNet',
    'vgg19': 'VGG',
    'vgg16': 'VGG',
    'densenet121': 'DenseNet'
}


def load_model(model_name, layer, dataset):
    """
    load model

    Args:
    :param: str(model_name)   # 'lenet', 'resnet18', 'resnet50', 'vgg19', 'densenet121'

    Return:
    :return: model
    """
    # _class = getattr(__import__('ialgebra.models'), name2class[model_name])
    _class = getattr(getattr(__import__("ialgebra"), "models"), name2class[model_name])

    # _class = getattr('models', name2class[model_name])
    if layer is not None:
        model = _class(name=model_name, dataset=dataset, layer=layer)
    else:
        model = _class(name=model_name, dataset=dataset)
    model = model.to(device)
    model.eval()
    if 'cuda' in device:
        _model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model, _model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)


def load_pretrained_model(model_name, layer, dataset, model_save_dir):
    """
    Function:
    :load pretrained model

    Args:
    :param: str(model_name)   # 'lenet', 'resnet18', 'resnet50', 'vgg19', 'densenet121'

    Inputs:
    :model_name
    :layer
    :dataset
    :model_dir

    Return:
    :return: model
    """
    _class = getattr(getattr(__import__("ialgebra"), "models"), name2class[model_name])
    if layer is not None:
        model = _class(name=model_name, dataset=dataset, layer=layer)
    else:
        model = _class(name=model_name, dataset=dataset)
    print("model_save_dir:", model_save_dir)
    # load pretrained weights
    assert os.path.isfile(model_save_dir), 'Error: no model checkpoint found!'
    # model_save_dir = os.path.join(model_dir, 'ckpt_{}_{}_{}.t7'.format(dataset, model_name, layer))
    model.load_state_dict(torch.load(model_save_dir))
    model = model.to(device)
    model.eval()
    freeze_model(model)  #freeze params' gradients
    return model


# Training
def train(epoch, net, trainloader, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 100 == 0:
            print('Epoch: %d, Step: [%d/%d], Loss: %.4f, Accuracy: %.3f%% (%d/%d)'
                    % (epoch+1, batch_idx+1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Testing
def test(args, epoch, net, testloader, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print('Epoch: %d, Step: [%d/%d], Loss: %.4f, Accuracy: %.3f%% (%d/%d)'
                        % (epoch+1, batch_idx+1, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        model_save_dir = os.path.join(args.model_dir, 'ckpt_{}_{}_{}.t7'.format(args.dataset, args.model_name, args.layer))
        torch.save(net.state_dict(), model_save_dir)
        best_acc = acc