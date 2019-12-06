import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from interpreters.interpreter_utils import resize_postfn, generate_map
from benchmark.data_utils import preprocess_fn



class SmoothGrad(Interpreter):

    def __init__(self, pretrained_model, device=None, stdev_spread=0.15, n_samples=25, magnitude=True):
        super(SmoothGrad, self).__init__()
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude


    def get_default_config():
        return dict(stdev_spread=0.15, n_samples=25, magnitude=True)


    def interpret_per_batch(self, index=None):
        x = self.preprocess_fn(self.bx).data.cpu().numpy()
        stdev = self.stdev_spread * (np.max(x, axis=0) - np.min(x, axis=0))
        total_gradients = np.zeros_like(x)

        for i in range(self.n_samples):
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
            output = self.pretrained_model(x_plus_noise)
            index = np.argmax(output.data.cpu().numpy()) if index is None else index
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
            one_hot = torch.sum(one_hot * output)

            if x_plus_noise.grad is not None:
                x_plus_noise.grad.data.zero_()
            one_hot.backward(retain_graph=True)
            grad = x_plus_noise.grad.data.cpu().numpy()
            total_gradients += (grad * grad) if self.magnitutde else grad

        avg_gradients = total_gradients[0, :, :, :] / self.n_samples   ##??
        avg_gradients = resize_postfn(avg_gradients)
        avg_gradients, avg_gradientsimg = generate_map(avg_gradients,self.bx)

        return avg_gradients, avg_gradientsimg 