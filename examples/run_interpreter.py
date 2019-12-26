#!/usr/bin/env python
# -*- coding: utf-8 -*-

# export PYTHONPATH=/home/hqs5468/workspace/Codes/projects/pytorch-ialgebra

import sys
import torch
import argparse
from ialgebra.interpreters import * 
from ialgebra.utils.utils_model import load_pretrained_model
from ialgebra.utils.utils_data import load_data,loader_to_tensor
from ialgebra.utils.utils_operation import ialgebra_interpreter, save_attribution_map, vis_saliancy_map
from matplotlib.pyplot import imshow


device = 'cuda' if torch.cuda.is_available() else 'cpu'

name2identity = {
    'grad_cam': 'GradCam',
    'grad_saliency': 'GradSaliency',
    'guided_backprop_grad': 'GuidedBackpropGrad',
    'guided_backprop_smoothgrad': 'GuidedBackpropSmoothGrad',
    'mask': 'Mask',
    'smoothgrad': 'SmoothGrad'
}


def main(config):

    dataset = config.dataset
    model_name = config.model_name
    layer = config.layer
    MODEL_PATH = config.model_path
    data_dir = config.data_dir
    batch_size = config.batch_size
    identity_name = config.identity_name

    # Generate Inputs
    trainloader = load_data(data_dir, dataset, 'train', batch_size = batch_size, shuffle=True)
    testloader = load_data(data_dir, dataset, 'test', batch_size = batch_size, shuffle=False)
    X_all, Y_all = loader_to_tensor(trainloader)
    bx = X_all[0:1]
    by = Y_all[0:1]

    # Load Model
    model_dir = MODEL_PATH + 'ckpt_' + dataset + '_' + model_name + '_' + layer + '.t7'
    model_kwargs = {'model_name': model_name, 'layer': layer, 'dataset': dataset, 'model_dir': MODEL_PATH}
    pretrained_model = load_pretrained_model(**model_kwargs)

    # define identity
    _identity_class = getattr(getattr(__import__("ialgebra"), "interpreters"), name2identity[identity_name])
    identity_interpreter = _identity_class(pretrained_model, dataset)

    # generate interpretation
    grad, gradimg = identity_interpreter(bx, by)
    vis_saliancy_map(grad, gradimg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Welcome to i-Algebra: An Interactive DNN Interpretater")
    parser.add_argument('--model_name', dest='model_name')
    parser.add_argument('--layer', dest='layer', default=None)
    parser.add_argument('--dataset', dest='dataset')
    parser.add_argument('--model_path', dest='model_path', default='/home/hqs5468/workspace/Codes/projects/pytorch-ialgebra/checkpoints/')
    parser.add_argument('--data_dir', dest='data_dir', default='/home/hqs5468/workspace/Codes/projects/pytorch-ialgebra/data/')
    parser.add_argument('--batch_size', dest='batch_size', default=128)
    parser.add_argument('--identity_name', dest='identity_name', default='smoothgrad') # ['grad_cam', 'grad_saliency', 'guided_backprop_grad', 'guided_backprop_smoothgrad', 'mask', 'smoothgrad']

    main(parser.parse_args())
