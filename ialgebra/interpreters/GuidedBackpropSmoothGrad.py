from ialgebra.interpreters.smoothgrad import SmoothGrad
from ialgebra.interpreters.GuidedBackpropGrad import GuidedBackpropReLU

class GuidedBackpropSmoothGrad(SmoothGrad):

    def __init__(self,pretrained_model=None, dataset=None):
        super(GuidedBackpropSmoothGrad, self).__init__(pretrained_model = pretrained_model, dataset=dataset, stdev_spread=.15, n_samples=25, magnitude=True)
        for idx, module in self.features._modules.items():
            if module.__class__.__name__ is 'ReLU':
                self.features._modules[idx] = GuidedBackpropReLU()