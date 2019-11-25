from interpreters.smoothgrad import SmoothGrad
from interpreters.GuidedBackpropGrad import GuidedBackpropReLU


class GuidedBackpropSmoothGrad(SmoothGrad):

    def __init__(self, pretrained_model, cuda=False, stdev_spread=.15, n_samples=25, magnitude=True):
        super(GuidedBackpropSmoothGrad, self).__init__(
            pretrained_model, cuda, stdev_spread, n_samples, magnitude)
        for idx, module in self.features._modules.items():
            if module.__class__.__name__ is 'ReLU':
                self.features._modules[idx] = GuidedBackpropReLU()

