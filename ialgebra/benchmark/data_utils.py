import os
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_name = {
    'mnist': 'MNIST',
    'cifar10': 'CIFAR10',
    'cifar100': 'CIFAR100',
    'imagenet': 'ImageNet',
}


def get_transform(dataset, mode):
    """
    generate transform for dataloader
    param
    :dataset: 'mnist', 'cifar10', 'cifar100', 'imagenet'
    :mode: 'train', 'test'
    return: transform
    """
    _transform = None
    if dataset == 'mnist' or 'cifar' in dataset:
        size = 28 if dataset == 'mnist' else 32
        if mode == 'train':
            _transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            _transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=3),
                transforms.ToTensor(),
            ])
    elif dataset == 'dogfish' or 'imagenet' in dataset:
        if mode == 'train':
            _transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            _transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
            ])
    return _transform



def load_data(data_dir, dataset, mode, self_data = False, batch_size=128, shuffle=False, num_workers=2):
    """
    Load data from the dir
    param
    :data_dir: the directory to store the data   # data_dir = os.path.join('/home/memory_data/', dataset, 'data')
    :dataset: {'mnist', 'cifar10', 'imagenet'}
    return: dataloader
    """
    _transform = get_transform(dataset, mode)
    if self_data:
        _dataset = datasets.ImageFolder(root=data_dir, transform=_transform)
    else:
        _dataset = getattr(datasets, dataset_name[dataset])
        kwargs = {'root': data_dir, 'download': True, 'transform': _transform}
        if dataset in ['imagenet']:
            kwargs['split'] = 'val' if mode == 'test' else 'train'
        else:
            kwargs['train'] = (mode == 'train')
        _dataset = _dataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader



def loader_to_tensor(dataloader):
    """
    convert loader to tensor
    param
    :dataloader: trainloader, testloader
    return: image_tensor, class_tensor
    """
    inputs_all = []
    targets_all = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_all.append(inputs)
        targets_all.append(targets)
    inputs_all = torch.cat(inputs_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    return inputs_all, targets_all






  




    





