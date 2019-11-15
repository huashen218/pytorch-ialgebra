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

sys.path.append('../ialgebra/')
from ialgebra.models import *
from ialgebra.benchmark.data_utils import load_data
from ialgebra.models.config_model import name2class, get_data_params

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0

def load_model(model, layer = None):
    """
    load model

    Args:
    :param: str(model_name)

    Return:
    :return: model
    """
    print('==> Building model..')
    if model == 'vgg19':
        net = VGG('VGG19')
    elif model == 'resnet18':
        net = ResNet18()
    elif model == 'resnet50':
        net = ResNet50()
    elif model == 'densenet121':
        net = DenseNet121()
    elif model == 'lenet':
        net = LeNet()
    elif model == 'd_resnet50':
        net = Dynamic_Model(layer)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return net


# ## revised   ???
# def load_model(model_name):
#     """
#     load model

#     Args:
#     :param: str(model_name)

#     Return:
#     :return: model
#     """
#     _class = getattr(__import__('model.' + name, fromlist=[name2class(name)]), name2class(name))
#     model = _class(name=args.model_name, data_dir=args.data_dir, dataset=args.dataset,
#                     layer=args.layer, **params)
#     model = model.to(device)
#     if 'cuda' in device:
#         model = torch.nn.DataParallel(model)
#         cudnn.benchmark = True
#     return model


# ## Ren
# def load_model(args):
#     params = parse_arguments(args)
#     name=get_model_name(args.model_name)
#     _class = getattr(__import__('model.' + name,
#                                 fromlist=[name2class(name)]), name2class(name))
#     _model = _class(name=args.model_name, data_dir=args.data_dir, dataset=args.dataset,
#                     layer=args.layer, **params)
#     _model.eval()
#     if torch.cuda.is_available():
#         _model = _model.cuda()
#     model = parallel_model(_model)

#     return _model, model


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
        state = {
            'net': net.state_dict(),
            'acc': acc,
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        model_save_dir = os.path.join(args.model_dir, 'ckpt_{}_{}_{}.t7'.format(args.dataset, args.model_name, args.layer))
        torch.save(state, model_save_dir)
        best_acc = acc