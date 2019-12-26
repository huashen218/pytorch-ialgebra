import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm

from ialgebra.utils.utils_interpreter import resize_postfn, generate_map
from ialgebra.utils.utils_data import preprocess_fn
from ialgebra.interpreters.interpreter import *
from ialgebra.utils.utils_data import to_numpy, to_tensor


class SmoothGrad(Interpreter):

    def __init__(self,pretrained_model=None, dataset=None, stdev_spread=0.15, n_samples=25, magnitude=True, index=None):
        super(SmoothGrad, self).__init__(pretrained_model = pretrained_model, dataset=dataset)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude
        self.index = index

    def interpret_per_batch(self, bxn, byn):
        x = self.preprocess_fn(bxn, self.dataset).data.cpu().numpy()
        stdev = self.stdev_spread * (np.max(x, axis=0) - np.min(x, axis=0))
        total_gradients = np.zeros_like(x)

        for i in tqdm(range(self.n_samples)):
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            x_plus_noise = Variable(torch.from_numpy(x_plus_noise).to(self.device), requires_grad=True)
            output = self.pretrained_model(x_plus_noise)
            index = np.argmax(output.data.cpu().numpy()) if self.index is None else self.index
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = Variable(torch.from_numpy(one_hot).to(self.device), requires_grad=True)
            one_hot = torch.sum(one_hot * output)

            if x_plus_noise.grad is not None:
                x_plus_noise.grad.data.zero_()
            one_hot.backward(retain_graph=True)
            grad = x_plus_noise.grad.data.cpu().numpy()
            total_gradients += (grad * grad) if self.magnitutde else grad

        avg_gradients = total_gradients[0, :, :, :] / self.n_samples   # shape [1, 3, 32, 32]
        avg_gradients = resize_postfn(to_tensor(avg_gradients))  # shape = [3, 32, 32]
        avg_gradients, avg_gradientsimg = generate_map(avg_gradients,bxn, self.dataset)

        return avg_gradients, avg_gradientsimg 
