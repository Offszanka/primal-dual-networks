"""A script which follows the instructions from Wang et al.

Implements the proximal deep structured model of the paper
S. Wang, S. Fidler, and R. Urtasun. Proximal Deep Structured Models. In D. Lee,
M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural In-
formation Processing Systems, volume 29. Curran Associates, Inc., 2016.
in short: PDSM
"""

import os
import warnings
import math

import torch
import torch.nn as nn

import models.reschapo as rcp



class Block(nn.Module):
    def __init__(self, W_kernel_size=3, V_kernel_size=3, y_size=2) -> None:
        super().__init__()
        self.cW = nn.Conv2d(
            1, y_size, W_kernel_size, padding=math.floor(W_kernel_size/2), stride=1, groups=1, bias=False)
        self.cSigma = nn.Parameter(torch.rand(1))

        self.cV = nn.Conv2d(
            y_size, 1, V_kernel_size, padding=math.floor(W_kernel_size/2), stride=1, groups=1, bias=False)

        self.prox_f = rcp.Prox_l1()
        self.prox_g = rcp.Prox_l2_learnable(tau=0.01*0.044)

        self.theta = nn.Parameter(torch.tensor(2., dtype=torch.float))

    def forward(self, inp):
        xk, yk, _xk, f = inp
        ykn = self.prox_f(yk + self.cW(_xk))
        xkn = self.prox_g(xk + self.cV(ykn), f)
        _xkn = xkn + self.theta*(xkn - xk)
        return xkn, ykn, _xkn, f


class RecurrentNet(rcp.PDNet):
    """Implements the proximal deep structured model of
        S. Wang, S. Fidler, and R. Urtasun. Proximal Deep Structured Models. In D. Lee,
        M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural In-
        formation Processing Systems, volume 29. Curran Associates, Inc., 2016.
    """
    def __init__(self, L: int = 1,
                 Vker_size: int =3, Wker_size: int=3, y_size: int=1,
                 use_same_block: bool=False,
                 use_cuda: bool=False,
                 load_path: os.PathLike=None
                 ):
        """
        Args:
            L (int, optional): Number of layers. Defaults to 1.
            Vker_size (int, optional): The kernel size of the matrix V. Defaults to 3.
            Wker_size (int, optional): The kernel size of the matrix W. Defaults to 3.
            y_size (int, optional): The channel size. Defaults to 1.
            use_same_block (bool, optional): Determines if the parameters should be shared. Defaults to False.
            use_cuda (bool, optional): If True sets the device to Cuda using the set_device method. Defaults to False.
            load_path (os.PathLike, optional): Deprecated - Path to load the data from. Defaults to None.
        """
        super().__init__()
        self.L = L
        self.y_size = y_size

        if use_same_block:
            self.block = Block(W_kernel_size=Wker_size,
                               V_kernel_size=Vker_size, y_size=y_size)
            self.network = nn.Sequential(*(L*[self.block]))
        else:
            layers = []
            for _ in range(L):
                layers.append(Block(W_kernel_size=Wker_size,
                              V_kernel_size=Vker_size, y_size=y_size))
            self.network = nn.Sequential(*layers)

        self.set_device(use_cuda)
        self.name = "PDSM"

        self.arguments = dict(
            L = L,
            Vker_size = Vker_size,
            Wker_size = Wker_size,
            y_size = y_size,
            use_same_block = use_same_block,
            use_cuda = use_cuda,
        )
        self.minfo = rcp.TabloidInfo(L, Vker_size, y_size, use_same_block)

    def forward(self, im):
        if self.device != im.device:
            im = im.to(self.device)

        xk = im
        _xk = im
        f = im.clone()
        b, _, h, w = im.shape
        yk = torch.zeros(b, self.y_size, h, w, device=self.device)
        _, _, _xkn, _ = self.network((xk, yk, _xk, f))
        return _xkn

    @torch.no_grad()
    def denoise(self, x):
        shape = x.shape
        shape = x.shape
        if len(x.shape) == 2:
            x = x[None, None]
        if len(x.shape) == 3:
            x = x[None]
        out = self(x.to(self.device))
        out.clip_(0., 1.)
        return out.reshape(shape).to(x.device)