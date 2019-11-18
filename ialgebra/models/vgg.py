from .model import *
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torchvision.models.vgg import model_urls

# todo
fc_dim_config = {
    'cifar10':  [64*7*7, 128*7*7, 256*7*7, 512*7*7, 512*7*7], 
    'imagenet': [64*7*7, 128*7*7, 256*7*7, 512*7*7, 512*7*7]
}

intervals = {
    "vgg16": [5, 10, 17, 24, 31],
    "vgg19": [5, 10, 19, 28, 37]
}


class VGG(Model):
    def __init__(self, name=None, dataset=None, layer=5, **kwargs):
        super(VGG, self).__init__(name = name, dataset = dataset, fc_depth=3, conv_dim = fc_dim_config[dataset][layer-1], fc_dim = 4096)
        self.layer = layer
        self.name = name
        self.interval = intervals[self.name]
        _model = models.__dict__[self.name](num_classes=self.num_classes)

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

    def load_official_weights(self):
        print("********Load From Official Website!********")
        _dict = model_zoo.load_url(model_urls[self.name])
        self.load_state_dict(_dict, strict=False)
        # self.features.load_state_dict(_dict, strict=False)
        # if self.num_classes == 1000:
        #     self.classifier.load_state_dict(_dict, strict=False)