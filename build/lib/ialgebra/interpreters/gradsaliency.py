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
