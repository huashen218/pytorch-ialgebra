import cv2
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ialgebra.interpreters.interpreter import *
from ialgebra.utils.utils_data import to_numpy, to_tensor


class FeatureExtractor(object):

    def __init__(self, model, target_layers):
        self.model = model
        self.features = model.features
        self.target_layers = target_layers
        self.gradients = []

    def __call__(self, x):
        target_activations, output = self.extract_features(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output

    def get_gradients(self):
        return self.gradients

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def extract_features(self, x):
        outputs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x




class GradCam(Interpreter):

    def __init__(self, pretrained_model=None, dataset=None, index=None,  target_layer_names = '4'):
        super(GradCam, self).__init__(pretrained_model = pretrained_model, dataset=dataset)
        self.extractor = FeatureExtractor(self.pretrained_model, target_layer_names)
        self.index= index

    # def get_default_gradcam_config(self):
    #     return dict(batch_size=1, target_layer_names = '3')

    def interpret_per_batch(self, bxn, byn):

        features, output = self.extractor(bxn)
        index = np.argmax(output.data.cpu().numpy()) if self.index is None else self.index

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = one_hot.to(self.device)
        one_hot = torch.sum(one_hot * output)

        self.pretrained_model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads = self.extractor.get_gradients()[-1].data.cpu().numpy()
        target = features[-1].data.cpu().numpy()[0, :]
        weights = np.mean(grads, axis=(2, 3))[0, :]
        cam = np.ones(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = self.resize_postfn(to_tensor(cam).unsqueeze(0))
        cam, camimg = self.generate_map(cam, bxn, self.dataset)

        return cam, camimg