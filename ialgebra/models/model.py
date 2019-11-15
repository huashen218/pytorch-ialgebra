# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

sys.path.append('../')
from benchmark.utils import to_tensor, to_numpy


global_config = {
    'mnist': {
        'mean': [0.1307, ],
        'std': [0.3081, ],
        'num_classes': 10,
        'n_dim': 28,
        'n_channel': 1
    },
    'cifar10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
        'num_classes': 10,
        'n_dim': 32,
        'n_channel': 3
    },
    'imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'num_classes': 1000,
        'n_dim': 224,
        'n_channel': 3
    },
    'none': {
        'mean': [0, 0, 0],
        'std': [1, 1, 1],
        'num_classes': 1,
    }
}



class Model(nn.Module):
    """Meta Model for DNN"""
    def __init__(self, name='abstact_model', dataset='none', num_classes=None, conv_depth=0, conv_dim=1, fc_depth=0, fc_dim=1, preprocess_ctrl = True, avgpool_ctrl = True, **kwargs):
        super(Model, self).__init__()
        self.dataset = dataset      # dataset = ['mnist', 'cifar10', 'imagenet', 'none']
        self.name = name  # model name
        self.config = global_config[self.dataset] 
        self.num_classes = self.config['num_classes']  # number of classes
        self.map_location = None if torch.cuda.is_available() else 'cpu'  # the location when loading pretrained weights using torch.load

        self.conv_depth = conv_depth
        self.conv_dim = conv_dim
        self.fc_depth = fc_depth
        self.fc_dim = fc_dim
        self.preprocess_ctrl = preprocess_ctrl
        self.avgpool_ctrl = avgpool_ctrl


        self.features = nn.Identity()   # feature extractor
        self.avgpool = nn.Identity()  # average pooling
        self.classifier = nn.Identity()  # classifier
        self.softmax = nn.Softmax(dim=1)

        if self.fc_depth > 0:
            seq = []
            if self.fc_depth == 1:
                seq.append(('fc', nn.Linear(self.conv_dim, self.num_classes)))
            else:
                seq.append(('fc1', nn.Linear(self.conv_dim, self.fc_dim)))
                seq.append(('relu1', nn.ReLU()))
                seq.append(('dropout1', nn.Dropout()))
                for i in range(self.fc_depth-2):
                    seq.append(
                        ('fc'+str(i+2), nn.Linear(self.fc_dim, self.fc_dim)))
                    seq.append(('relu'+str(i+2), nn.ReLU()))
                    seq.append(('dropout'+str(i+2), nn.Dropout()))
                seq.append(('fc'+str(self.fc_depth),
                            nn.Linear(self.fc_dim, self.num_classes)))
            self.classifier = nn.Sequential(OrderedDict(seq))


    # forward method
    # input: (batch_size, channels, height, width)
    # output: (batch_size, logits)
    def forward(self, x):
        # if x.shape is (channels, height, width)
        # (channels, height, width) ==> (batch_size: 1, channels, height, width)
        if len(x.shape) == 3:
            x.unsqueeze_(0) 
        x = self.preprocess(x) if self.preprocess_ctrl else x     # module: preprocess
        x = self.get_fm(x)                                        # module: feature
        x = self.avgpool(x) if self.avgpool_ctrl else x           # module: avgpool
        x = torch.flatten(x, 1)
        x = self.get_logits_from_fm(x)                            # module: classifier
        return x


    # This is defined by Pytorch documents
    # See https://pytorch.org/docs/stable/torchvision/models.html for more details
    # The input range is [0,1]
    # input: (batch_size, channels, height, width)
    # output: (batch_size, channels, height, width)
    def preprocess(self, x):
        mean = to_tensor(self.config['mean'])
        std = to_tensor(self.config['std'])
        return x.sub(mean[None, :, None, None]).div(std[None, :, None, None])


    # get feature map
    # input: (batch_size, channels, height, width)
    # output: (batch_size, [feature_map])
    def get_fm(self, x):
        return self.features(x)


    # get logits from feature map
    # input: (batch_size, [feature_map])
    # output: (batch_size, logits)
    def get_logits_from_fm(self, x):
        return self.classifier(x)




    # ##########################  To Do  ##########################
    # # weights_loc: (default: None) if None, use the default path. Else if the path doesn't exist, quit.
    # # full: (default: False) whether save feature extractor.
    # # output: (default: False) whether output help information.

    # def load_pretrained_weights(self, weights_loc=None, full=False, output=False):
    #     if output:
    #         print("********Pretrained %s Loaded!********" % self.name)

    #     if weights_loc is None:
    #         weights_loc = self.data_dir + \
    #             '%s/model/%s.pth' % (self.dataset, self.name)
    #         if os.path.exists(weights_loc):
    #             if output:
    #                 print("********Load From Saved File: %s********" % weights_loc)
    #             self.load_state_dict(torch.load(weights_loc, map_location=self.map_location))
    #         else:
    #             print('Default model file not exist: ', weights_loc)
    #             self.load_official_weights()
    #     elif os.path.exists(weights_loc):
    #         if output:
    #             print("********Load From Saved File: %s********" % weights_loc)
    #         if full:
    #             self.load_state_dict(
    #                 torch.load(weights_loc, map_location=self.map_location))
    #         else:
    #             self.classifier.load_state_dict(
    #                 torch.load(weights_loc, map_location=self.map_location))
    #     else:
    #         print('File not Found: ', weights_loc)
    #         self.load_official_weights()

    # # weights_loc: (default: None) if None, use the default path.
    # # full: (default: False) whether save feature extractor.
    # def save_weights(self, weights_loc='', full=False):
    #     if weights_loc == '':
    #         model_dir = self.data_dir + '%s/model/' % self.dataset
    #         if not os.path.exists(model_dir):
    #             os.makedirs(model_dir)
    #         weights_loc = model_dir+'%s.pth' % (self.name)
    #         torch.save(self.state_dict(), weights_loc)
    #     else:
    #         if full:
    #             torch.save(self.state_dict(), weights_loc)
    #         else:
    #             torch.save(self.classifier.state_dict(), weights_loc)


    # # define in concrete model class.
    # def load_official_weights(self):
    #     print("Nothing Happens. No official pretrained data to load.")
