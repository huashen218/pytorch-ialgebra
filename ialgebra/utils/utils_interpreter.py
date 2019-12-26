import cv2
import torch
import numpy as np
import torch.nn.functional as F

from ialgebra.interpreters import * 
from ialgebra.utils.utils_model import load_pretrained_model
from ialgebra.utils.utils_data import to_numpy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

name2identity = {
    'grad_cam': 'gradcam.GradCam',
    'grad_saliency': 'gradsaliency.GradSaliency',
    'guided_backprop_grad': 'GuidedBackpropGrad.GuidedBackpropGrad',
    'guided_backprop_smoothgrad': 'GuidedBackpropSmoothGrad.GuidedBackpropSmoothGrad',
    'mask': 'mask.Mask',
    'smoothgrad': 'smoothgrad.SmoothGrad'
}

map_size = {
    'cifar10': 32,
    'imagenet': 224
}


def generate_identity(bx, by, model_kwargs, identity_name, identity_kwargs):
    """
    *Function*: 
    :generate saliency_map with same shape of input

    *Inputs*:
    :input: shape = [B*C*H*W], type = torch.Tensor()
    :model_kwargs: pretrain_dnn    model_params = {'model_name': model_name, 'layer': layer, 'dataset': dataset, 'model_dir': model_dir}
    :identity_name:   from list = ['cam', 'grad_cam', 'grad_saliency', 'mask', 'rts']
    :identity_kwargs: param in interpretaiton_methods

    *Returns*:
    :identity_map: shape = [B*C*H*W], type = numpy.ndarray
    :identity_mapimg: int_map+img (might not use): shape = [B*C*H*W], type = numpy.ndarray 
    """
    pretrained_model = load_pretrained_model(**model_kwargs)
    _identity_class = getattr(getattr(__import__("ialgebra"), "interpreters"), name2identity[identity_name])
    # identity_map, identity_mapimg = _identity_class(pretrained_model, bx, by, device)       ## initialize the interpreter (pretrained_model, bx, by, device, preprocess_fn = preprocess_fn, resize_postfn = resize_postfn, generate_map = generate_maps, batch_size = 1)
    identity_interpreter = _identity_class(pretrained_model, bx, by, device)       ## initialize the interpreter (pretrained_model, bx, by, device, preprocess_fn = preprocess_fn, resize_postfn = resize_postfn, generate_map = generate_maps, batch_size = 1)
    return identity_interpreter



# [N, 3, 224, 224] -> [N, 112, 112]
def resize_postfn(grad):
    grad = grad.unsqueeze(0) if len(grad.size()) == 3 else grad   # ensure len(grad.size()) == 4
    grad = grad.abs().max(1, keepdim=True)[0] if grad.size(1)==3 else grad     # grad shape = [1, 3, 32, 32]
    grad = F.avg_pool2d(grad, 4).squeeze(1)        # grad shape = [1, 1, 32, 32]
    shape = grad.shape                             # grad shape = [1, 8, 8]
    grad = grad.view(len(grad), -1)
    grad_min = grad.min(1, keepdim=True)[0]
    grad = grad - grad_min
    grad_max = grad.max(1, keepdim=True)[0]
    grad = grad / torch.max(grad_max, torch.tensor([1e-8], device='cuda'))
    return grad.view(*shape)


def normalize_map(maps):
    shape = maps.shape
    n = len(maps)
    flatten_map = maps.reshape((n, -1))
    m_min, m_max = np.min(flatten_map, axis=1, keepdims=True), np.percentile(flatten_map, 99, axis=1, keepdims=True)
    normed_maps = np.clip((flatten_map - m_min) / (m_max - m_min), 0, 1)
    return normed_maps.reshape(shape)



# from [112, 112] -> [3, 224, 224] map
def generate_map(grad, img, dataset):
    mask = grad.cpu().detach().numpy()
    mask = mask.squeeze(0)  
    
    smap =  normalize_map(mask[None])[0]
    smap = np.uint8(255 * cv2.resize(np.repeat(smap, 3, 0), (map_size[dataset], map_size[dataset]), interpolation=cv2.INTER_LINEAR))
    heatmap = cv2.applyColorMap(smap, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255   # heatmap_shape = [map_size, map_size, 3]
    heatmap = heatmap[:,:,::-1].transpose(2,0,1)
    cam = heatmap + np.float32(to_numpy(img.squeeze(0)))  # img_shape = [3, map_size, map_size]
    cam = cam / np.max(cam)
    # imshow(cam[:,:,::-1])
    # cv2.imwrite(save_dir, cam*255)
    return heatmap, cam





    