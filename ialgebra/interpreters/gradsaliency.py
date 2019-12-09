#!/usr/bin/env python
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from ialgebra.interpreters.interpreter import *
from torch.autograd import Variable

class GradSaliency(Interpreter):
    def __init__(self, pretrained_model=None, dataset=None):
        super(GradSaliency, self).__init__(pretrained_model = pretrained_model, dataset=dataset)

    def interpret_per_batch(self, bxn, byn):
        bxp = self.preprocess_fn(bxn, self.dataset)
        logit = self.pretrained_model(bxp)
        loss = F.nll_loss(F.log_softmax(logit), byn)
        grad = autograd.grad([loss], [bxn], create_graph=False)[0]
        grad = self.resize_postfn(grad)
        grad, gradimg = self.generate_map(grad, bxn, self.dataset)
        return grad, gradimg




class VanillaGrad(Interpreter):
    def __init__(self, pretrained_model):
        self.pretrained_model = pretrained_model
        self.features = pretrained_model.features
        self.pretrained_model.eval()

    def __call__(self, x, index=None):
        output = self.pretrained_model(x)
        index = np.argmax(output.data.cpu().numpy()) if index is None else index
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)
        grad = x.grad.data.cpu().numpy()
        grad = grad[0, :, :, :]
        return grad
