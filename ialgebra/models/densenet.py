import torch.nn as nn
from collections import OrderedDict
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torchvision.models.densenet import model_urls

from ialgebra.models import *

fc_dim_config = {
    # 'cifar10': [128*4*4, 256*2*2, 512, 1024], 
    'cifar10': [128, 256, 512, 1024], 
    'imagenet':[128*28*28, 256*14*14, 512*7*7, 1024*7*7]
}

class DenseNet(Model):
    def __init__(self, name=None, dataset=None, layer = '4', **kwargs):
        super(DenseNet, self).__init__(name = name, dataset = dataset, fc_depth=1, conv_dim = fc_dim_config[dataset][int(layer)-1], avgpool_ctrl = False)
        self.layer = int(layer)
        self.name = name
        _model = models.__dict__[self.name](num_classes=self.num_classes)


        if self.dataset == 'imagenet':
          pre_feature = [('conv0', _model.features.conv0), ('norm0', _model.features.norm0), ('relu0', _model.features.relu0), ('pool0', _model.features.pool0)]
          feature = []
          for i in range(self.layer):
            feature.append(('denseblock'+str(i+1), getattr(_model.features, 'denseblock'+str(i+1))))
            if i < 3:
              feature.append(('transition'+str(i+1), getattr(_model.features, 'transition'+str(i+1))))
          if self.layer == 4:
            feature.append(('norm5',  _model.features.norm5))
          features = pre_feature+feature
          self.features = nn.Sequential(OrderedDict(features))

        elif self.dataset == 'cifar10':
          pre_feature = [
            ('conv0', nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)), 
            ('norm0', _model.features.norm0), 
            ('relu0', _model.features.relu0), 
            ('pool0', _model.features.pool0)]
          feature = []
          for i in range(self.layer):
            feature.append(('denseblock'+str(i+1), getattr(_model.features, 'denseblock'+str(i+1))))
            if i < 3:
              feature.append(('transition'+str(i+1), getattr(_model.features, 'transition'+str(i+1))))
          if self.layer == 4:
            feature.append(('norm5',  _model.features.norm5))
          features = pre_feature+feature
          self.features = nn.Sequential(OrderedDict(features))
          self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def load_official_weights(self):
        print("********Load From Official Website!********")
        _dict = model_zoo.load_url(model_urls[self.name])
        self.load_state_dict(_dict, strict=False)
        # self.features.load_state_dict(_dict, strict=False)
        # if self.num_classes == 1000:
        #     self.classifier.load_state_dict(_dict, strict=False)