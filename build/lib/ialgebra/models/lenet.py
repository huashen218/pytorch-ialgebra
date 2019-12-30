# -*- coding: utf-8 -*-
from .model import *
import torch.nn as nn
from collections import OrderedDict


fc_dim_config = {
    'mnist':  [20*12*12, 50*4*4]
}

class LeNet(Model):
    def __init__(self, name='lenet', dataset='mnist', layer='2', **kwargs):
        super(LeNet, self).__init__(name = name, dataset = dataset, fc_depth=2, conv_dim = fc_dim_config[dataset][int(layer)-1], fc_dim = 500, avgpool_ctrl = False)
        self.layer = int(layer)
        self.name = name

        lenet = {
            "layer1": [
            ('conv1', nn.Conv2d(1, 20, 5, 1)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('relu1', nn.ReLU())],
            "layer2": [
            ('conv2', nn.Conv2d(20, 50, 5, 1)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('relu2', nn.ReLU())]
        }

        feature = []
        for k in range(self.layer):
            feature += lenet["layer"+str(k+1)]
        self.features = nn.Sequential(OrderedDict(feature))