#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import visdom
import argparse
import numpy as np
import matplotlib.image as mpimg

from ialgebra.operators import *
from ialgebra.utils.utils_main import read_config
from ialgebra.utils.utils_model import load_model
from ialgebra.utils.utils_operation import ialgebra_interpreter, save_attribution_map, vis_saliancy_map


def main(config):

  MODEL_PATH = '/home/hqs5468/workspace/Codes/projects/pytorch-ialgebra/checkpoints/'
  model_dir = MODEL_PATH + 'ckpt_' + config.dataset + '_' + config.model_name + '_' + config.layer + '.t7'
  print("model_dir:", model_dir)
  model_params = {'model_name': config.model_name, 'layer': config.layer, 'dataset': config.dataset, 'model_dir': model_dir}





    # step2: invoke ialgebra interpreter
    compositer = False if len(operators) == 1 else True
    ialgebra_map, ialgebra_mapimg = ialgebra_interpreter(inputs_tup, models_tup, identity, identity_kwargs, operators_tup, operators_kwargs_tup, compositer = compositer)

    # step3: save or visualization
    vis_saliancy_map(ialgebra_map, ialgebra_mapimg)                  # visualization
    save_attribution_map(ialgebra_map, ialgebra_mapimg, save_dir)    # save 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Welcome to i-Algebra: An Interactive DNN Interpretater")
    # parser.add_argument('--config', '-c', required=False, dest='config_path', default='./declarative_query_example.yaml')

    parser.add_argument('--model_name', dest='model_name')
    parser.add_argument('--layer', dest='layer', default=None)
    parser.add_argument('--dataset', dest='dataset')



    args = parser.parse_args()
    main(args.config_path)