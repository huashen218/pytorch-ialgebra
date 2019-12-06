# -*- coding: utf-8 -*-
import cv2
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np


# map = {'int': torch.int, 'float': torch.float,
#        'double': torch.double, 'long': torch.long}

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









# def normalize_map(maps):
#     shape = maps.shape
#     n = len(maps)
#     flatten_map = maps.reshape((n, -1))
#     m_min, m_max = np.min(flatten_map, axis=1, keepdims=True), np.percentile(flatten_map, 99, axis=1, keepdims=True)
#     normed_maps = np.clip((flatten_map - m_min) / (m_max - m_min), 0, 1)
#     return normed_maps.reshape(shape)


# # to revise
# def normalize_map(maps, map_size = None):
#     """
#     Normalize the map

#     Inputs: 2d or 3d numpy map;

#     Outputs:
#     3D normalized map [0,1]
#     """
#     shape = maps.shape
#     n = len(maps)
#     # # 3D map
#     # if n == 3:
#     #     if maps.shape[0] != 3:
#     #         maps = maps.transpose(2, 0, 1)  # maps shape [3, map_size, map_size]


#     # elif n == 2:
#     # #

#     flatten_map = maps.reshape((n, -1))
#     m_min, m_max = np.min(flatten_map, axis=1, keepdims=True), np.percentile(flatten_map, 99, axis=1, keepdims=True)
#     normed_maps = np.clip((flatten_map - m_min) / (m_max - m_min), 0, 1)
#     return normed_maps.reshape(shape)


# def plot(img_2d, img, in_size = 224):
#     img_2d = np.sum(img, axis=0)
#     span = abs(np.percentile(img_2d, percentile=99))
#     vmin = -span
#     vmax = span
#     img_2d = np.clip((img_2d - vmin) / (vmax - vmin), -1, 1)
#     imshow(img_2d)
    
#     map = np.uint8(255 * cv2.resize(np.repeat(img_2d, 3, 0), (in_size, in_size), interpolation=cv2.INTER_LINEAR))
#     map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
#     imshow((img[:,:,::-1]+map[:,:,::-1]/255) / (img[:,:,::-1]+map[:,:,::-1]/255).max())
    
# def imagenet_resize_postfn(grad):
#     grad = grad.abs().max(1, keepdim=True)[0]
#     grad = F.avg_pool2d(grad, 4).squeeze(1)
#     shape = grad.shape
#     grad = grad.view(len(grad), -1)
#     grad_min = grad.min(1, keepdim=True)[0]
#     grad = grad - grad_min
#     grad_max = grad.max(1, keepdim=True)[0]
#     grad = grad / torch.max(grad_max, torch.tensor([1e-8], device='cuda'))
#     return grad.view(*shape)


# def save_as_gray_image(img, filename, percentile=99):
#     img_2d = np.sum(img, axis=0)
#     span = abs(np.percentile(img_2d, percentile))
#     vmin = -span
#     vmax = span
#     img_2d = np.clip((img_2d - vmin) / (vmax - vmin), -1, 1)
#     cv2.imwrite(filename, img_2d * 255)

#     return


# def save_cam_image(img, mask, filename):
#     heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     cam = np.uint8(255 * cam)
# #     cv2.imwrite(filename, cam)


