# -*- coding: utf-8 -*-
data_configs = {
    'mnist': {
        'n_dim': 28,
        'n_channel': 1,
        'num_classes': 10,
        'model_name': 'lenet'
    },
    'cifar10': {
        'n_dim': 32,
        'n_channel': 3,
        'num_classes': 10,
        'model_name': 'resnetnew18'
    },
    'imagenet': {
        'n_dim': 224,
        'n_channel': 3,
        'num_classes': 1000,
        'model_name': 'resnet'
    },
    'sample_imagenet': {
        'n_dim': 224,
        'n_channel': 3,
        'num_classes': 1000,
        'model_name': 'resnet18'
    },
}

name2class_map = {
    'lenet': 'LeNet',
    'vgg': 'VGG',
    'resnet18': 'ResNet18',
    'resnet50': 'ResNet50',
    'densenet121': 'DenseNet121'
}

def get_data_params(dataset):
    return data_configs[dataset]

def name2class(name):
    return name2class_map[name]
