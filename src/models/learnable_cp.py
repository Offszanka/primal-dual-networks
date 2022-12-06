# -*- coding: utf-8 -*-
"""
Implements Learnable Chambolle-Pock (LCP).
"""

import torch
import torch.nn as nn

import models.reschapo as rcp
from PyChaPo import _gradient, dis_divergence


class PyCPBlock(nn.Module):
    """Implements the building blocks of the LCP network.
    """

    def __init__(self, sigma=None, tau=0.01, lam=0.03) -> None:
        super().__init__()
        if sigma is None:
            sigma = 1/(tau*8)
        self.prox_f = rcp.Prox_l1(sigma=1.)
        self.prox_g = rcp.Prox_l2_learnable(tau=tau*lam)
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True)
        self.tau = nn.Parameter(torch.tensor(tau), requires_grad=True)
        self.theta = nn.Parameter(torch.tensor(2.), requires_grad=True)

    def forward(self, inp):
        xk, yk, _xk, f = inp
        ykn = self.prox_f(yk + self.sigma*_gradient(_xk))
        xkn = self.prox_g(xk + self.tau*dis_divergence(ykn), f)
        _xkn = xkn + self.theta*(xkn - xk)
        return xkn, ykn, _xkn, f


class LCP(rcp.PDNet):
    """Implements LCP, the learned Chambolle-Pock network.

    Here, the single parameters of the Chambolle-Pock iteration are learned.

    Args:
        sigma (float, optional): The sigma parameter of the Chambolle-Pock
            algorithm. Defaults to 1/(tau*8).
        tau (float, optional): The tau parameter of the Chambolle-Pock algorithm. Defaults to 0.01.
        lam (float, optional): the lambda. Defaults to 0.03.
        max_iter (int, optional): Maximal number of iterations. Defaults to 150.
        use_cuda (bool, optional): If True sets the device on the cuda device by
            using set_device of rcp.PDNet. Defaults to True.
        use_same_block (bool, optional): Determines if the parameters
            should be shared. Defaults to False.

    """

    def __init__(self, sigma: float = None, tau: float = 0.01, lam: float = 0.03, max_iter: int = 30, use_cuda: bool = True, use_same_block: bool = False) -> None:
        super().__init__()
        if sigma is None:
            sigma = 1/(tau*8)
        self.max_iter = int(max_iter)

        if use_same_block:
            self.block = PyCPBlock(sigma=sigma, tau=tau, lam=lam)
            self.network = nn.Sequential(*(self.max_iter*[self.block]))
        else:
            self.network = nn.Sequential(
                *[PyCPBlock(sigma=sigma, tau=tau, lam=lam) for _ in range(self.max_iter)])

        self.set_device(use_cuda)
        self.name = "Classic learnable Chambolle-Pock"
        self.arguments = dict(
            max_iter=max_iter,
            tau=tau,
            lam=lam,
            sigma=sigma,
            use_same_block=use_same_block,
            use_cuda=use_cuda,
        )
        self.minfo = rcp.TabloidInfo(max_iter, None, None, use_same_block)

    def forward(self, im):
        if im.device != self.device:
            im = im.to(self.device)
        xk = im
        _xk = im
        f = im.clone()
        b, c, h, w = im.shape
        yk = torch.zeros(b, 2*c, h, w, device=self.device)
        _, _, _xkn, _ = self.network((xk, yk, _xk, f))
        return _xkn

    @torch.no_grad()
    def denoise(self, im):
        if self.device != im.device:
            im_device = im.device
            im = im.to(self.device)
        sh = im.shape
        if len(im.shape) == 2:
            # Expected: im.shape = (h,w)
            im = im[None, None, :, :]
        elif len(im.shape) == 3:
            # Expected: im.shape = (c,h,w)
            im = im[None, :, :]
        it = self(im)
        return it.reshape(sh).detach().to(im_device).clip(0, 1)
