#!/usr/bin/env python
# -*- coding: utf-8 -*-

# export PYTHONPATH=/home/hqs5468/workspace/Codes/projects/pytorch-ialgebra

import sys
import torch
import argparse
from ialgebra.interpreters import * 
from ialgebra.utils.utils_main import read_config
from ialgebra.utils.utils_model import load_pretrained_model
from ialgebra.utils.utils_data import load_data,loader_to_tensor
from ialgebra.utils.utils_operation import ialgebra_interpreter, save_attribution_map, vis_saliancy_map
from matplotlib.pyplot import imshow
from ialgebra.operations.operator import Operator
from ialgebra.operations.compositer import Compositer


ialgebra_key = {
    'operator': Operator,
    'compositer': Compositer
}


#######################################################################
def main(config_path):

    # step1: parsing
    config = read_config(config_path)
    
    # data
    dataset = config['data']['name']
    subname = config['data']['subname']
    indexes = config['data']['idx_list']
    data_path = config['data']['data_path']
    batch_size = config['data']['batch_size']

    # models
    model_list= list(config['models'].values())
    print("model_list:", model_list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # identity
    identity_name = config['identity']['identity_name']
    identity_kwarg = config['identity']['identity_kwarg'] if config['identity']['identity_kwarg']!='None' else dict()

    # ialgebra
    (key, value), = config['ialgebra'].items()
    ialgebra_name, ialgebra_kwarg = value['name'], value['kwarg']
    if key == 'operator':
        ialgebra_name = value['name']
        ialgebra_kwargs = value['kwarg']
    if key == 'compositer':
        ialgebra_name = value['name']
        ialgebra_kwargs = value['kwarg']

    save_path = config['save_path']
    print("save_path:", save_path)


    # Load Data
    print("============Loading Dataset============")
    trainloader = load_data(data_path, dataset, 'train', batch_size = batch_size, shuffle=True)
    testloader = load_data(data_path, dataset, 'test', batch_size = batch_size, shuffle=False)
    X_all, Y_all = loader_to_tensor(trainloader) if subname == 'train' else loader_to_tensor(testloader)

    bx_list, by_list = [X_all[indexes[0]].unsqueeze(0), X_all[indexes[1]].unsqueeze(0)], [Y_all[indexes[0]].unsqueeze(0), Y_all[indexes[1]].unsqueeze(0)]

    # Load Model
    print("============Loading Model============")
    pretrained_models = []
    for k in range(len(model_list)):
        model_dir = model_list[k]['model_path'] + 'ckpt_' + dataset + '_' + model_list[k]['name'] + '_' + model_list[k]['layer'] + '.t7'
        print("model_dir:", model_dir)
        model_kwargs = {'model_name': model_list[k]['name'], 'layer': model_list[k]['layer'], 'dataset': dataset, 'model_save_dir': model_dir}
        pretrained_models.append(load_pretrained_model(**model_kwargs))

    # Load Operator
    print("============Loading Operator============")
    _operation_class = Operator(identity_name, dataset, device = device) if key == 'operator' else Compositer(identity_name, dataset, device = device)
    operation = getattr(_operation_class, ialgebra_name)


    ###################### To customize ######################
    heatmap1, heatmapimg1= operation(bx_list[0], by_list[0], pretrained_models[0], **ialgebra_kwargs)

    # visualize and save image
    vis_saliancy_map(heatmap1, heatmapimg1)
    # vis_saliancy_map(heatmap2, heatmapimg2)
    save_attribution_map(heatmap1, heatmapimg1, save_path) 
    ###################### To customize ######################



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Welcome to i-Algebra: An Interactive DNN Interpretater")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='./declarative_query_example.yaml')
    args = parser.parse_args()
    main(args.config_path)