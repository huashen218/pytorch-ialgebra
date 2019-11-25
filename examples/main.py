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
from ialgebra.benchmark.utils import read_config, 
from ialgebra.operators import *
from ialgebra.benchmark.model_utils import load_model
from ialgebra.operations.utils_operation import ialgebra_interpreter, save_attribution_map, vis_saliancy_map



def main(config_path):
    """Functions:
    1) parse user config;
    2) invode ialgebra interpreter
    3) visualize the 'img+saliency_map'

    Inputs: 
    :config_path of declarative_query.yaml
      :input: [B*C*H*W]
      :model: pretrained_model
    """
    config = read_config(config_path)

    # step1: parsing
    """
    parsing: 
    """

    inputs = config['input_lists']
    input_size = inputs.shape(3)
    models = config['model_lists']
    identity = config['interpreters']['identity']
    declarative_query = config['interpreters']['declarative_query']
    save_dir = config['save_dir'] #todo parse declarative_query to operators and compositions
    operators = []

    # step2: invoke ialgebra interpreter
    compositer = False if len(operators) == 1 else True
    ialgebra_map, ialgebra_mapimg = ialgebra_interpreter(inputs_tup, models_tup, identity, identity_kwargs, operators_tup, operators_kwargs_tup, compositer = compositer)

    # step3: save or visualization
    vis_saliancy_map(ialgebra_map, ialgebra_mapimg)                  # visualization
    save_attribution_map(ialgebra_map, ialgebra_mapimg, save_dir)    # save 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Welcome to i-Algebra: An Interactive DNN Interpretater")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='./declarative_query_example.yaml')
    args = parser.parse_args()
    main(args.config_path)