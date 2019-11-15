# -*- coding: utf-8 -*-

import sys
import yaml
import torch
import torch.nn as nn
import numpy as np



map = {'int': torch.int, 'float': torch.float,
       'double': torch.double, 'long': torch.long}

_cuda = torch.cuda.is_available()


def read_config(config_path):
    """read the config parameters in declarative_query.yaml file

    Args:
        :param config_path

    Return: 
        :parsed config
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config

    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        exit(-1)



def to_tensor(x, dtype=None, device=None, copy=False):
    if x is None:
        return None
    _dtype = map[dtype] if isinstance(dtype, str) else dtype
    _device = device.device if torch.is_tensor(device) else device

    if isinstance(x, list):
        try:
            x = torch.stack(x)
        except Exception:
            pass
    try:
        x = torch.as_tensor(x, dtype=_dtype, device=_device)
    except Exception:
        print('tensor: ', x)
        raise ValueError()
    if _device is None and _cuda and not x.is_cuda:
        x = x.cuda()
    return x.contiguous()


def to_numpy(x):
    if x is None:
        return None
    if type(x).__module__ == np.__name__:
        return x
    if torch.is_tensor(x):
        return (x.cpu() if x.is_cuda else x).detach().numpy()
    return np.array(x)