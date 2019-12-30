import torch
import numpy as np
from ialgebra.utils.utils_data import preprocess_fn
from ialgebra.utils.utils_interpreter import resize_postfn, generate_map

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Interpreter(object):
    def __init__(self, pretrained_model=None,  dataset=None, target_layer=None):

        super(Interpreter, self).__init__()
        self.pretrained_model = pretrained_model
        self.dataset = dataset
        self.features = self.pretrained_model.features
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.resize_postfn = resize_postfn
        self.generate_map = generate_map
        self.target_layer = target_layer

    def __call__(self, bx, by, batch_size=None):
        return self.interpret(bx, by, batch_size=batch_size)

    def get_default_config(self):
        pass

    def interpret_per_batch(self, bxn, byn):
        pass

    def interpret(self, bx, by, batch_size = None):
        print("bx:", bx.size())
        print("by,", by.size())
        bx = bx.unsqueeze(0) if len(bx.size()) != 4 else bx          # assert len(x.size) = 4, bx torch.Tensor()
        batch_size = len(bx) if batch_size is None else batch_size
        n_batches = (len(bx) + batch_size - 1) // batch_size
        interpreter_map = []
        interpreter_mapimg = []
        for i in range(n_batches):
            si = i * batch_size
            ei = min(len(bx), si + batch_size)
            bxn, byn = bx[si:ei], by[si:ei]
            bxn, byn = torch.tensor(bxn, requires_grad=True), torch.tensor(byn)
            gradmap, gradimg = self.interpret_per_batch(bxn, byn)
            interpreter_map.append(gradmap)
            interpreter_mapimg.append(gradimg)
        interpreter_map = np.concatenate(interpreter_map, axis=0)
        interpreter_mapimg = np.concatenate(interpreter_mapimg, axis=0)
        return interpreter_map, interpreter_mapimg
