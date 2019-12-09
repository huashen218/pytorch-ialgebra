import torch.nn as nn
from collections import OrderedDict
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls

from ialgebra.models import *

fc_dim_config = {
    "resnet18": {
        'cifar10': [64, 128, 256, 512], 
        'imagenet':[64, 128, 256, 512]
    },
    "resnet50": {
        'cifar10': [256, 512, 1024, 2048], 
        'imagenet':[256, 512, 1024, 2048]
    }
}

class ResNet(Model):
    def __init__(self, name=None, dataset=None, layer = '4', **kwargs):
        super(ResNet, self).__init__(name = name, dataset = dataset, fc_depth=1, conv_dim = fc_dim_config[name][dataset][int(layer)-1])
        self.layer = int(layer)
        self.name = name
        self.dataset = dataset
        _model = models.__dict__[self.name](num_classes=self.num_classes)

        if self.dataset == 'imagenet':
            feature = ([
                ('conv1', _model.conv1), 
                ('bn1', _model.bn1), 
                ('relu', _model.relu), 
                ('maxpool', _model.maxpool)] + 
                [('layer'+str(i+1), getattr(_model, 'layer'+str(i+1))) for i in range(self.layer)])
            self.features = nn.Sequential(OrderedDict(feature))
            self.avgpool = _model.avgpool

        elif self.dataset == 'cifar10':
            feature = ([
                ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)), 
                ('bn1', _model.bn1), 
                ('relu', _model.relu), 
                ('maxpool', _model.maxpool)] + 
                [('layer'+str(i+1), getattr(_model, 'layer'+str(i+1))) for i in range(self.layer)])
            self.features = nn.Sequential(OrderedDict(feature))
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def load_official_weights(self):
        print("********Load From Official Website!********")
        _dict = model_zoo.load_url(model_urls[self.name])
        self.load_state_dict(_dict, strict=False)
        # self.features.load_state_dict(_dict, strict=False)
        # if self.num_classes == 1000:
        #     self.classifier.load_state_dict(_dict, strict=False)