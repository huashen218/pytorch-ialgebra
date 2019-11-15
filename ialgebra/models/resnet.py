from model import *
from torchvision.models.resnet import model_urls
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
import torch.utils.model_zoo as model_zoo

fc_dim_config = {
    'cifar10': [256*26*26, 512*10*10, 1024*2*2, 2048*4*4], 
    'imagenet':[256, 512, 1024, 2048],
    'none': [0, 0, 0, 0]
}

class ResNet(Model):
    def __init__(self, name='resnet50', dataset='none', layer = 4, **kwargs):
        super(ResNet, self).__init__(fc_depth=1, conv_dim = fc_dim_config[dataset][layer-1])
        self.layer = layer
        self.name = name
        _model = models.__dict__[self.name](num_classes=self.num_classes)

        feature = ([('conv1', _model.conv1), ('bn1', _model.bn1), ('relu', _model.relu), ('maxpool', _model.maxpool)] + 
        [('layer'+str(i+1), getattr(_model, 'layer'+str(i+1))) for i in range(self.layer)])
        self.features = nn.Sequential(OrderedDict(feature))
        self.avgpool = _model.avgpool 


    # def load_official_weights(self):
    #     print("********Load From Official Website!********")
    #     _dict = model_zoo.load_url(model_urls[self.name])
    #     self.features.load_state_dict(_dict, strict=False)
    #     if self.num_classes == 1000:
    #         self.classifier.load_state_dict(_dict, strict=False)
