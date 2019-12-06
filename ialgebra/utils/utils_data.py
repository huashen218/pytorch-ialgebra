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



preprocess_img_config = {
    'cifar10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
    },

    'imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },

    'other': {
        'mean': [0, 0, 0],
        'std': [1, 1, 1]
    }
}


def to_tensor(x):
    return x.to(device) if isinstance(x, torch.Tensor) else torch.Tensor(x).to(device)

def to_numpy(x):
    return x if isinstance(x, np.ndarray) else (x.cpu() if x.is_cuda else x).detach().numpy()


def preprocess_fn(x, dataset):
    """
    preprocess data before feeding into model
    """
    ts = [torch.unsqueeze((x[:, i] - preprocess_img_config[dataset]['mean'][i]) / preprocess_img_config[dataset]['std'][i], 1) for i in range(3)]
    return torch.cat(ts, dim=1)


def _get_transform(dataset, mode):
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
                # transforms.RandomCrop(32, padding=4),
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
    _transform = _get_transform(dataset, mode)
    if self_data:
        _dataset = datasets.ImageFolder(root=data_dir, transform=_transform)  # data_dir -> class images
    elif dataset in ['sample_imagenet']:
        datadir = '/home/hqs5468/workspace/Codes/projects/pytorch-ialgebra/data/sample_imagenet/data/sample_imagenet/'
        _dataset = datasets.ImageFolder(root=datadir+mode, transform=_transform)
    elif dataset == 'imagenet':
        _dataset = getattr(datasets, dataset_name[dataset])
        data_dir = '/home/rbp5354/workspace/data/imagenet/data'
        kwargs = {'root': data_dir, 'download': True, 'transform': _transform}
        kwargs['split'] = 'val' if mode == 'test' else 'train'
        _dataset = _dataset(**kwargs)
    else:
        _dataset = getattr(datasets, dataset_name[dataset])
        kwargs = {'root': data_dir, 'download': True, 'transform': _transform}
        kwargs['train'] = (mode == 'train')
        _dataset = _dataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print("dataloader:", dataloader)
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




# def to_tensor(x, dtype=None, device=None, copy=False):
#     if x is None:
#         return None
#     _dtype = map[dtype] if isinstance(dtype, str) else dtype
#     _device = device.device if torch.is_tensor(device) else device

#     if isinstance(x, list):
#         try:
#             x = torch.stack(x)
#         except Exception:
#             pass
#     try:
#         x = torch.as_tensor(x, dtype=_dtype, device=_device)
#     except Exception:
#         print('tensor: ', x)
#         raise ValueError()
#     if _device is None and _cuda and not x.is_cuda:
#         x = x.cuda()
#     return x.contiguous()


# def to_numpy(x):
#     if x is None:
#         return None
#     if type(x).__module__ == np.__name__:
#         return x
#     if torch.is_tensor(x):
#         return (x.cpu() if x.is_cuda else x).detach().numpy()
#     return np.array(x)
