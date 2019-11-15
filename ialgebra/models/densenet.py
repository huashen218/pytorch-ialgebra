from model import *
from torchvision.models.resnet import model_urls
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
import torch.utils.model_zoo as model_zoo


fc_dim_config = {
    'cifar10': [0, 0, 0, 0], 
    'imagenet':[0, 0, 0, 0],
    'none': [0, 0, 0, 0]
}

class DenseNet(Model):
    def __init__(self, name='densenet50', dataset='none', layer = 4, **kwargs):
        super(DenseNet, self).__init__(fc_depth=1, conv_dim = fc_dim_config[dataset][layer-1], avgpool_ctrl = False)
        self.layer = layer
        self.name = name
        _model = models.__dict__[self.name](num_classes=self.num_classes)

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


    # def load_official_weights(self):
    #     print("********Load From Official Website!********")
    #     _dict = model_zoo.load_url(model_urls[self.name])
    #     self.features.load_state_dict(_dict, strict=False)
    #     if self.num_classes == 1000:
    #         self.classifier.load_state_dict(_dict, strict=False)
