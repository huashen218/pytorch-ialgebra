import sys
import torch
import torchvision
import torch.nn as nn

sys.path.append('../')
from models import *

### This is only for Cifar10, ResNet50

class Dynamic_Model(nn.Module):
    def __init__(self, layer, num_classes=10):
        super(Dynamic_Model, self).__init__()
        self.init_model()
        self.layer = layer
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.num_classes = num_classes

        # # only ResNet 50 Cifar10
        # if self.layer == 'layer1':
        #     self.num = 256
        # elif self.layer == 'layer2':
        #     self.num = 512
        # elif self.layer == 'layer3':
        #     self.num = 1024
        # elif self.layer == 'layer4':
        #     self.num = 2048

        # # only ResNet 50 Cifar10
        # if self.layer == 'layer1':
        #     self.num = 256*26*26
        # elif self.layer == 'layer2':
        #     self.num = 512*10*10
        # elif self.layer == 'layer3':
        #     self.num = 1024*2*2
        # elif self.layer == 'layer4':
        #     self.num = 2048*4*4

        # self.fc = nn.Linear(self.num, self.num_classes)

    def init_model(self):
        self.model = ResNet50()

    def linear_layer(self, o):
        x = self.avgpool(o[0]) if o[0].size(2) >7 else o[0]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_layer(self, x):
        o = self.model.get_layer(x, self.layer)
        return o

    def forward(self, x):
        o = self.get_layer(x)

        # get fc input plane
        o_num = self.avgpool(o[0]) if o[0].size(2) >7 else o[0]
        self.num = o_num.size(1) * o_num.size(2) *o_num.size(3)
        
        self.fc = nn.Linear(self.num, self.num_classes)
        # o = self.fc(o)

        o = self.linear_layer(o)




        return o