import torch
import numpy as np
from benchmark.data_utils import preprocess_fn
from interpreters.interpreter_utils import resize_postfn, generate_map

class Interpreter(object):
    
    def __init__(self, name = 'abstract_interpreter', pretrained_model, bx, by, device, preprocess_fn = preprocess_fn, resize_postfn = resize_postfn, generate_map = generate_maps, batch_size = 1):
        super(Interpreter, self).__init__()
        self.name = name
        self.pretrained_model = pretrained_model
        self.features = self.pretrained_model.features
        self.preprocess_fn = preprocess_fn
        self.bx = bx
        self.by = by
        self.device = device
        self.resize_postfn = resize_postfn
        self.generate_map = generate_map
        self.batch_size = batch_size

    def __call__(self):
        return interpret()

    def get_default_config(self):
        pass

    def interpret_per_batch(self, **kwargs):
        pass

    def interpret(self, **kwargs):
        n_batches = (len(x) + self.batch_size - 1) // self.batch_size
        interpreter_map = []
        interpreter_mapimg = []
        for i in range(n_batches):
            si = i * self.batch_size
            ei = min(len(x), si + self.batch_size)
            bx, by = x[si:ei], y[si:ei]
            bx, by = torch.tensor(bx, requires_grad=True), torch.tensor(by)
            grad, gradimg = self.interpret_per_batch(self.pretrained_model, self.preprocess_fn, self.bx, self.by, **kwargs)
            interpreter_map.append(grad)
            interpreter_mapimg.append(gradimg)
        interpreter_map = np.concatenate(interpreter_map, axis=0)
        interpreter_mapimg = np.concatenate(interpreter_mapimg, axis=0)
        return interpreter_map, interpreter_mapimg
