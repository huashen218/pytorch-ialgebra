from ialgebra.models import *
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torchvision.models.vgg import model_urls


fc_dim_config = {
    # 'cifar10':  [64*7*7, 128*7*7, 256*7*7, 512*7*7, 512*7*7], 
    # 'cifar10':  [256*8*8, 128*8*8, 64*8*8, 32*8*8, 16*8*8], 
    'cifar10':  [256*8*8, 128*8*8, 64*8*8, 32*8*8, 16*8*8], 
    'imagenet': [64*7*7, 128*7*7, 256*7*7, 512*7*7, 512*7*7]
}

intervals = {
    "vgg16": [5, 10, 17, 24, 31],
    "vgg19": [5, 10, 19, 28, 37]
}

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(Model):
    def __init__(self, name=None, dataset=None, layer='5', **kwargs):
        super(VGG, self).__init__(name = name, dataset = dataset, fc_depth=3, conv_dim = fc_dim_config[dataset][int(layer)-1], fc_dim = 4096, **kwargs)
        self.layer = int(layer)
        self.name = name
        self.dataset = dataset
        self.interval = intervals[self.name]
        _model = models.__dict__[self.name](num_classes=self.num_classes)

        if self.dataset == 'cifar10':
            self.features = self._make_layers(cfg[self.name])
            self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
            # self.classifier = nn.Linear(512, 10)

        elif self.dataset == 'imagenet':
            vgg = {
                "layer1": [(str(i), _model.features[i]) for i in range(self.interval[0])],
                "layer2": [(str(i), _model.features[i]) for i in range(self.interval[0], self.interval[1])],
                "layer3": [(str(i), _model.features[i]) for i in range(self.interval[1], self.interval[2])],
                "layer4": [(str(i), _model.features[i]) for i in range(self.interval[2], self.interval[3])],
                "layer5": [(str(i), _model.features[i]) for i in range(self.interval[3], self.interval[4])]}
            feature = []
            for k in range(self.layer):
                feature += vgg["layer"+str(k+1)]
            self.features = nn.Sequential(OrderedDict(feature))
            self.avgpool = _model.avgpool


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        layer_cnt = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if layer_cnt == self.layer:
                    return nn.Sequential(*layers)
                else:
                    layer_cnt += 1
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x



    def load_official_weights(self):
        print("********Load From Official Website!********")
        _dict = model_zoo.load_url(model_urls[self.name])
        self.load_state_dict(_dict, strict=False)
        # self.features.load_state_dict(_dict, strict=False)
        # if self.num_classes == 1000:
        #     self.classifier.load_state_dict(_dict, strict=False)