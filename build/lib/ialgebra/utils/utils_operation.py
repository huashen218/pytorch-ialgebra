import os
import numpy as np
import visdom
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
import warnings
warnings.filterwarnings("ignore")



# parse declarative query to ialgebra_name
name2operator = {
    'projection': 'operator.projection',
    'selection': 'operator.selection',
    'join': 'operator.join',
    'antijoin': 'operator.antijoin'
}

name2compositer = {
    'slct_proj': 'compositer.slct_proj',
    'slct_join': 'compositer.slct_join',
    'slct_anjo': 'compositer.slct_anjo',
    'proj_join': 'compositer.proj_join',
    'proj_anjo': 'compositer.proj_anjo',
    'slct_proj_join': 'compositer.slct_proj_join',
    'slct_proj_anjo': 'compositer.slct_proj_anjo',
}


def ialgebra_interpreter(inputs_tup,  models_tup, identity_name, operators_tup, operators_kwargs_tup, compositer, identity_kwargs=None):
    """
    *Function*: 
    operate saliency_maps to meet user demands
    None: if there are multiple inputs, but only one is needed to interpret for the operation, we choose the 1st input as default.


    *Inputs*:
    :input_tup: ((bx1, by1), (bx2,by2),...,(bxn,byn))
    :models_tup: (model1_kwargs, model2_kwargs, ..., modeln_kwargs)
               :  model_kwargs: {'model_name': model_name, 'layer': layer, 'dataset': dataset, 'model_dir': model_dir}
    :identity_name: str()
    :identity_kwargs: dict{}
    :operators: (operator1, operator2 ,..., operatorn)
    :operators_kwargs: (operator1_kwargs, operator2_kwargs ,..., operatorn_kwargs)
               : operator_kwargs: dict{}


    *Returns*:
    :ialgebra_map: shape = [B*C*H*W], type = numpy.ndarray
    :ialgebra_mapimg: ialgebra_map+img (might not use): shape = [B*C*H*W], type = numpy.ndarray 
    """
    # parse ialgebra_name:
    ialgebra_name = operators_tup  #todo

    if compositer:
        _ialgebra_class = getattr(getattr(__import__("ialgebra"), "operations"), name2compositer[ialgebra_name]) 
    else:
        _ialgebra_class = getattr(getattr(__import__("ialgebra"), "operations"), name2operator[ialgebra_name]) 
    
    ialgebra_map, ialgebra_mapimg = _ialgebra_class(inputs_tup,  models_tup, identity_name, operators_tup, operators_kwargs_tup, identity_kwargs)
    # to revise

    return ialgebra_map, ialgebra_mapimg



def vis_saliancy_map(ialgebra_map, ialgebra_mapimg, vis_method = 'imshow', vis_sport = 80):
    """
    Visualize attribution map

    Input: 
    :ialgebra_map: shape = [B*C*H*W]
    :ialgebra_mapimg:  ialgebra_map+img (shape = [B*C*H*W])
    :vis_method = 'imshow', 'vis'
    """

    if vis_method == 'imshow':
        if len(ialgebra_mapimg.shape) == 3:
            imshow(ialgebra_mapimg.transpose(1,2,0))   # imshow (H, W, C)
        elif len(ialgebra_mapimg.shape) == 4:
            for k in range(len(ialgebra_mapimg)):
                imshow(ialgebra_mapimg[k][:,:,::-1])

    elif vis_method == 'visdom':
        vis = visdom.Visdom(env='ialgebra_visualization', port=vis_sport)
        # vis.images([ialgebra_map, ialgebra_mapimg], win='iAlgebra_maps', opts=dict(title='iAlgebra_mapimg'))


def save_attribution_map(ialgebra_map, ialgebra_mapimg, save_dir, save_ialgebra_map = False):
    """
    Save attribution map:
    : 1) .npz dicts
    : 2) .png figures

    Input: 
    :ialgebra_map: shape = [B*C*H*W]
    :ialgebra_mapimg:  ialgebra_map+img (shape = [B*C*H*W])
    :save_dir
    """

    # save .npy file
    save_dobj = {'ialgebra_map': ialgebra_map, 'ialgebra_mapimg': ialgebra_mapimg}
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    np.savez(os.path.join(save_dir, 'ialgebra_maps.npz'), **save_dobj)

    # save figures
    if len(ialgebra_mapimg.shape) == 3:
        mpimg.imsave(os.path.join(save_dir, 'ialgebra_mapimg_None.png'), np.transpose(ialgebra_mapimg, (1, 2, 0)))
    elif len(ialgebra_mapimg.shape) == 4:
        for k in range(len(ialgebra_mapimg)):
            mpimg.imsave(os.path.join(save_dir, 'ialgebra_mapimg_{}.png'.format(k)), np.transpose(ialgebra_mapimg[k], (1, 2, 0)))
    
    if save_ialgebra_map:
        if len(ialgebra_mapimg.shape) == 3:
            mpimg.imsave(os.path.join(save_dir, 'ialgebra_map_None.png'), np.transpose(ialgebra_map, (1, 2, 0)))
        elif len(ialgebra_mapimg.shape) == 4:
            for k in range(len(ialgebra_map)):
                mpimg.imsave(os.path.join(save_dir, 'ialgebra_map_{}.png'.format(k)), np.transpose(ialgebra_map[k], (1, 2, 0)))
    