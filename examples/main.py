#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import visdom
import argparse
sys.path.append('../ialgebra/')
import numpy as np
import matplotlib.image as mpimg
from benchmark.utils import read_config



def operator(model, input, ex_name):
    """
    Implements the operators and composition to generate saliency map as .npy

    Input:
    :model: pre_trained model
    :input: shape [B*C*H*W]
    :ex_name: interpreter_name (e.g., 'gradient_saliency’)
    :


    Return:
    

    """
    model_tup = load_model(config)

    dobj = np.load(config.data_path)
    img_x, img_y, img_yt = dobj['img_x'], dobj['img_y'], dobj['img_yt']
    benign_gs = generate_gs(model_tup, img_x, img_y, imagenet_resize_postfn, False, batch_size=50)
    save_dobj = {'img_x': img_x, 'img_y': img_y, 'benign_gs': benign_gs, 'img_yt': img_yt}
    np.savez(config.save_path, **save_dobj)


    return saliency_map








def normalize_map(maps):
    """
    Normalize the map
    """

    shape = maps.shape
    n = len(maps)
    flatten_map = maps.reshape((n, -1))
    m_min, m_max = np.min(flatten_map, axis=1, keepdims=True), np.percentile(flatten_map, 99, axis=1, keepdims=True)
    normed_maps = np.clip((flatten_map - m_min) / (m_max - m_min), 0, 1)
    return normed_maps.reshape(shape)


def saliancy_map_img(smap, img, in_size, save_dir):
    """
    Generate saliency map from .npy to .png

    Input: 
    :smap: map [1*H*W]
    :img: raw img  [B*C*H*W]
    :in_size: dim of img [e.g., 32]
    :save_dir
    
    Return:
    :interpreted_img 
    """
    smap = normalize_map(smap[None])[0]
    map = np.uint8(255 * cv2.resize(np.repeat(smap, 3, 0), (in_size, in_size), interpolation=cv2.INTER_LINEAR))
    map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
    map = np.float32(map / 255.).transpose([2, 0, 1])[::-1]
    interpreted_img = (img + map)
    interpreted_img = interpreted_img / interpreted_img.max()
    smap = np.repeat(smap[None], 3, 0)

    # save img
    mpimg.imsave(save_dir, np.transpose(img, (1, 2, 0)))
    # vis with visdom
    vis = visdom.Visdom(env='ialgebra', port=config.port)
    vis.images([img, smap, interpreted_img], win='window_name', opts=dict(title='Base'))

    return interpreted_img












def main(config_path):
    """main

    Args:
        :config_path of declarative_query.yaml

    Return: 
    """
    config = read_config(config_path)
    print("Declarative Query: ", config)

    # parse declarative query
    inputs = config['input_lists']
    models = config['model_lists']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Welcome to i-Algebra: An Interactive DNN Interpretater")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='./declarative_query_example.yaml')
    args = parser.parse_args()
    main(args.config_path)