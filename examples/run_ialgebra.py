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
from ialgebra.operations.operator import * 
from ialgebra.operations.compositer import * 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(config):

    dataset = config.dataset
    model_name = config.model_name
    layer = config.layer
    MODEL_PATH = config.model_path
    data_dir = config.data_dir
    batch_size = config.batch_size
    identity_name = config.identity_name
    operator_name = config.operator_name
    compositer_name = config.compositer_name

    # Generate Inputs
    trainloader = load_data(data_dir, dataset, 'train', batch_size = batch_size, shuffle=True)
    testloader = load_data(data_dir, dataset, 'test', batch_size = batch_size, shuffle=False)
    X_all, Y_all = loader_to_tensor(trainloader)

    # Load Model
    model_dir = MODEL_PATH + 'ckpt_' + dataset + '_' + model_name + '_' + layer + '.t7'
    model_kwargs = {'model_name': model_name, 'layer': layer, 'dataset': dataset, 'model_dir': MODEL_PATH}
    pretrained_model = load_pretrained_model(**model_kwargs)

    # define operator
    _operator_class = Operator(identity_name, dataset, device = device)
    operator = getattr(_operator_class, operator_name)


    # # one model, multi models;  one input, multiple inputs
    # bx = X_all[0:2]
    # by = Y_all[0:2]
    # model_list = pretrained_model
    # heatmap1, heatmapimg1, heatmapimg2  = operator(bx, by, model_list)


    # define compositer
    _compositer_class = Compositer(identity_name, dataset, device = device)
    compositer = getattr(_compositer_class, compositer_name)    
    
    # bx, by = X_all[0:1], Y_all[0:1]
    # bx_list, by_list = X_all[0:2], Y_all[0:2]
    # model_list = [pretrained_model1, pretrained_model2]
    # model = pretrained_model1
    # region1, region2 = [5,15,15,25], [5,15,15,25]
    # heatmap1, heatmapimg1, heatmap2, heatmapimg2 = compositer(bx_list, by_list, model_list, region1, region2)


    # visualize and save image
    # vis_saliancy_map(heatmap1, heatmapimg1)
    # vis_saliancy_map(heatmap2, heatmapimg2)
    # save_attribution_map(grad, gradimg, './')    # save 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Welcome to i-Algebra: An Interactive DNN Interpretater")
    parser.add_argument('--model_name', dest='model_name')
    parser.add_argument('--layer', dest='layer', default=None)
    parser.add_argument('--dataset', dest='dataset')
    parser.add_argument('--model_path', dest='model_path', default='/home/hqs5468/workspace/Codes/projects/pytorch-ialgebra/checkpoints/')
    parser.add_argument('--data_dir', dest='data_dir', default='/home/hqs5468/workspace/Codes/projects/pytorch-ialgebra/data/')
    parser.add_argument('--batch_size', dest='batch_size', default=128)
    parser.add_argument('--identity_name', dest='identity_name', default='smoothgrad') # ['grad_cam', 'grad_saliency', 'guided_backprop_grad', 'guided_backprop_smoothgrad', 'mask', 'smoothgrad']
    parser.add_argument('--operator_name', dest='operator_name', default='selection') # 
    parser.add_argument('--compositer_name', dest='compositer_name', default='slct_proj') # 

    main(parser.parse_args())
