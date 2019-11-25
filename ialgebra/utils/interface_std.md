



**input**





**model**
def load_pretrained_model(model_name, layer, dataset, model_dir):
    """
    Function:
    :load pretrained model

    Args:
    :param: str(model_name)   # 'lenet', 'resnet18', 'resnet50', 'vgg19', 'densenet121'

    Inputs:
    :model_name
    :layer
    :dataset
    :model_dir

    Return:
    :return: model

    output = logits
    """


**interpreters**
def generate_identity_map(input, model_kwargs, interpreter_name, interpreter_kwargs):
    """
    *Function*: 
    :generate saliency_map with same shape of input

    *Inputs*:
    :input: shape = [B*C*H*W], type = torch.Tensor().to(device)  # note, tensor+device
    :model_kwargs: pretrain_dnn    model_params = {'model_name': model_name, 'layer': layer, 'dataset': dataset, 'model_dir': model_dir}
    :interpreter_name:   from list = ['cam', 'grad_cam', 'grad_saliency', 'mask', 'rts']
    :interpreter_kwargs: param in interpretaiton_methods

    *Returns*:
    :int_map: shape = [B*C*H*W], type = numpy.ndarray
    :interpreter_mapimg: int_map+img (might not use): shape = [B*C*H*W], type = numpy.ndarray 
    """


**operations**
def generate_ialgebra_map(input, model_kwargs, identity_name, identity_kwargs, ialgebra_name, ialgebra_kwargs):
    """
    *Function*: 
    operate saliency_maps to meet user demands

    *Inputs*:
    :input:
    :model_kwargs:
    :identity_name:  set default if no user inputs
    :identity_kwargs:
    :ialgebra_name:
    :ialgebra_kwargs:

    *Returns*:
    :ialgebra_map: shape = [B*C*H*W], type = numpy.ndarray
    :ialgebra_mapimg: ialgebra_map+img (might not use): shape = [B*C*H*W], type = numpy.ndarray 
    """





    ## note: x, y algready torch.Tensor().to(device)
    ## note: pretrained_model: logits layer