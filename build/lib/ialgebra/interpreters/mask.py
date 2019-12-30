import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from ialgebra.interpreters.interpreter import *



map_size = {
    'cifar10': 32,
    'imagenet': 224
}


class _GaussianBlur(nn.Module):
    def __init__(self, ksize, sigma, num_channels=3):
        super(_GaussianBlur, self).__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.psize = int((ksize - 1) / 2)
        self.num_channels = num_channels
        self.blur_kernel = nn.Parameter(self.get_gaussian_blur_kernel().repeat(num_channels, 1, 1, 1), requires_grad=False)


    def get_gaussian_blur_kernel(self):
        ker = cv2.getGaussianKernel(self.ksize, self.sigma).astype(np.float32)
        blur_kernel = (ker * ker.T)[None, None]
        blur_kernel = torch.tensor(blur_kernel)
        return blur_kernel


    def forward(self, x):
        x_padded = F.pad(x, [self.psize] * 4, mode="reflect")
        return F.conv2d(x_padded, self.blur_kernel, groups=self.num_channels)



class Mask(Interpreter):

    def __init__(self, pretrained_model=None, dataset=None, mask_config = None, m_init=None):
        super(Mask, self).__init__(pretrained_model = pretrained_model, dataset=dataset)
        self.blur1 = _GaussianBlur(21, -1)
        self.blur2 = _GaussianBlur(11, -1, 1)
        self.mask_config = mask_config
        self.m_init = m_init
        self.blur1.to(self.device)
        self.blur2.to(self.device)
        self.map_size = map_size[self.dataset]



    def get_default_config(self):
        return dict(lr=0.1, l1_lambda=1e-4, tv_lambda=1e-2, noise_std=0, n_iters=400, batch_size=40, verbose=False)


    def tv_norm(self, img, beta=2., epsilon=1e-8):
        batch_size = img.size(0)
        dy = -img[:, :, :-1] + img[:, :, 1:]
        dx = (img[:, :, :, 1:] - img[:, :, :, :-1]).transpose(2, 3)
        return (dx.pow(2) + dy.pow(2) + epsilon).pow(beta / 2.).view(batch_size, -1).sum(1)


    def mask_iter(self, x, y, m_init, l1_lambda=1e-4, tv_lambda=1e-2, tv_beta=3., noise_std=0., weights=None, x_blurred=None):
        batch_size = x.size(0)
        x_blurred = self.blur1(x) if x_blurred is None else x_blurred

        jitter = x_blurred.size(2)-self.map_size
        j1, j2 = (np.random.randint(jitter), np.random.randint(jitter)) if jitter != 0 else (0, 0)
        # x_ = x[:, :, j1:j1+self.map_size, j2:j2+self.map_size]
        x_ = x
        x_blurred_ = x_blurred[:, :, j1:j1+self.map_size, j2:j2+self.map_size]

        if noise_std != 0:
            noisy = torch.randn_like(m_init)
            mask_w_noisy = m_init + noisy
            mask_w_noisy.clamp_(0, 1)
        else:
            mask_w_noisy = m_init

        mask_w_noisy = F.interpolate(mask_w_noisy, (self.map_size, self.map_size), mode='bilinear')
        mask_w_noisy = self.blur2(mask_w_noisy)
        x = x_ * mask_w_noisy + x_blurred_ * (1 - mask_w_noisy)

        class_loss = F.softmax(self.pretrained_model(self.preprocess_fn(x, self.dataset)), dim=-1).gather(1, y.unsqueeze(1)).squeeze(1)
        l1_loss = (1 - m_init).abs().view(batch_size, -1).sum(-1)
        tv_loss = self.tv_norm(m_init, tv_beta)

        if weights is None:
            tot_loss = l1_lambda * torch.sum(l1_loss) + tv_lambda * torch.sum(tv_loss) + torch.sum(class_loss)
        else:
            tot_loss = (l1_lambda * torch.sum(l1_loss * weights) + tv_lambda * torch.sum(tv_loss * weights) +
                        torch.sum(class_loss * weights))
        return tot_loss, [l1_loss, tv_lambda, class_loss]


    def interpret_per_batch(self, bxn, byn):
        mask_config = self.get_default_config() if self.mask_config is None else self.mask_config

        batch_size = len(bxn)
        m_init = torch.zeros(batch_size, 1, 8, 8).fill_(0.5) if self.m_init is None else self.m_init.clone().detach()
        m_init = m_init.to(self.device)
        m_init.requires_grad = True

        optimizer = Adam([m_init], lr=mask_config['lr'])
        bx = F.interpolate(bxn, (self.map_size + 2, self.map_size + 2), mode='bilinear')
        bx_blurred = self.blur1(bx)
        for i in range(mask_config['n_iters']):
            tot_loss = self.mask_iter(bxn, byn, m_init, mask_config['l1_lambda'], mask_config['tv_lambda'],
                                    noise_std=mask_config['noise_std'], x_blurred=bx_blurred)[0]
            if mask_config['verbose'] and i % 50 == 0:
                print(i, np.asscalar(tot_loss) / batch_size)
            optimizer.zero_grad()
            tot_loss.backward(retain_graph=True)
            optimizer.step()
            m_init.data.clamp_(0, 1)
        
        m_init = self.resize_postfn(m_init)
        m_init, m_initimg = self.generate_map(m_init, bxn, self.dataset)    # todo: batch
        return m_init, m_initimg

