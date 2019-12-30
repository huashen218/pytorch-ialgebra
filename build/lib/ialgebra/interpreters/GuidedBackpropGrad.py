import torch
from ialgebra.interpreters.gradsaliency import GradSaliency



class GuidedBackpropGrad(GradSaliency):

    def __init__(self, pretrained_model=None, dataset=None):
        super(GuidedBackpropGrad, self).__init__(pretrained_model = pretrained_model, dataset=dataset)
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()
        for idx, module in self.features._modules.items():
            if module.__class__.__name__ is 'ReLU':
                self.features._modules[idx] = GuidedBackpropReLU()



class GuidedBackpropReLU(torch.autograd.Function):

    def __init__(self, inplace=False):
        super(GuidedBackpropReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        pos_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, pos_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        pos_mask_1 = (input > 0).type_as(grad_output)
        pos_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, pos_mask_1), pos_mask_2)
        return grad_input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'


