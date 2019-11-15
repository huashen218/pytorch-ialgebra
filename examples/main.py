#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import sys 
sys.path.append('../ialgebra/')
from benchmark.utils import read_config



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