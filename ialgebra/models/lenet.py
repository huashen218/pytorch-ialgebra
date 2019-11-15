# -*- coding: utf-8 -*-

from model import *
from torchvision.models.resnet import model_urls
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
import torch.utils.model_zoo as model_zoo



# todo
fc_dim_config = {
    'mnist':  [0, 0, 0, 0, 0], 
    'none':     [0, 0, 0, 0, 0]
}


class LeNet(Model):
    def __init__(self, name='lenet', dataset='mnist', layer=2, **kwargs):
        super(LeNet, self).__init__(fc_depth=2, conv_dim = fc_dim_config[dataset][layer-1], fc_dim = 500, avgpool_ctrl = False)
        self.layer = layer
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



    # def load_official_weights(self):
    #     print("********Load From Official Website!********")
    #     _dict = model_zoo.load_url(model_urls[self.name])
    #     self.features.load_state_dict(_dict, strict=False)
    #     if self.num_classes == 1000:
    #         self.classifier.load_state_dict(_dict, strict=False)
