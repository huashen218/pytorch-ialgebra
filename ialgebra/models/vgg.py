from model import *
from torchvision.models.resnet import model_urls
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
import torch.utils.model_zoo as model_zoo

# todo
fc_dim_config = {
    'cifar10':  [0, 0, 0, 0, 0], 
    'imagenet': [0, 0, 0, 0, 0],
    'none':     [0, 0, 0, 0, 0]
}

intervals = {
    "vgg16": [5, 10, 17, 24, 31],
    "vgg19": [5, 10, 19, 28, 37]
}


class VGG(Model):
    def __init__(self, name='vgg19', dataset='none', layer=5, **kwargs):
        super(VGG, self).__init__(fc_depth=3, conv_dim = fc_dim_config[dataset][layer-1], fc_dim = 4096)
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



    # def load_official_weights(self):
    #     print("********Load From Official Website!********")
    #     _dict = model_zoo.load_url(model_urls[self.name])
    #     if self.num_classes==1000:
    #         self.load_state_dict(_dict)
    #     else:
    #         new_dict = OrderedDict()
    #         for name, param in _dict.items():
    #             if 'classifier.6' not in name:
    #                 new_dict[name] = param
    #         self.load_state_dict(new_dict, strict=False)
