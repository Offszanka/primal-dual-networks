"""
This file implements the residual block proposed by
C. Brauer and D. Lorenz. Primal-dual residual networks, 2018

In short:
PDResNet
"""

import math
import warnings

import torch.nn as nn

import models.reschapo as rcp


class Block(nn.Module):
    def __init__(self, W_kernel_size=3, V_kernel_size=3, c_exp=2, inc=1, outc=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=inc, out_channels=c_exp*inc,
            kernel_size=W_kernel_size, padding=math.floor(W_kernel_size/2), bias=True,
            groups=inc)

        self.conv2 = nn.Conv2d(
            in_channels=c_exp*inc, out_channels=outc,
            kernel_size=V_kernel_size, padding=math.floor(V_kernel_size/2), bias=False,
        )

        self.prox_F = rcp.Prox_l1()
        self.prox_G = rcp.Prox_l2_learnable(tau=1e-5)

    def forward(self, input):
        xk, f = input
        out = self.conv1(xk)
        out = self.prox_F(out)

        out = self.conv2(out)
        out = out + xk
        out = self.prox_G(out, f)

        return out, f


class ResidualNet(rcp.PDNet):
    """A class to represent the Primal-Dual residual network proposed by
    C. Brauer and D. Lorenz. Primal-dual residual networks, 2018
    """

    def __init__(self, L: int = 1,
                 Vker_size: int = 3, Wker_size: int = 3, c_exp: int = 1,
                 use_same_block: bool = False,
                 use_cuda: bool = False):
        """
        Args:
            L (int, optional): Number of layers. Defaults to 1.
            Vker_size (int, optional): The kernel size of the V matrix. Defaults to 3.
            Wker_size (int, optional): The kernel size of the V matrix. Defaults to 3.
            c_exp (int, optional): Expanse factor of the channel size. Defaults to 1.
            use_same_block (bool, optional): Determines if the parameters should be shared. Defaults to False.
            use_cuda (bool, optional): If True sets the device on the cuda device by using set_device of rcp.PDNet. Defaults to False.
        """
        super().__init__()
        self.L = L
        self.c_exp = c_exp

        if use_same_block:
            self.block = Block(W_kernel_size=Wker_size,
                               V_kernel_size=Vker_size, c_exp=c_exp)
            self.network = nn.Sequential(*(L*[self.block]))
        else:
            layers = []
            for _ in range(L):
                layers.append(Block(W_kernel_size=Wker_size,
                                    V_kernel_size=Vker_size, c_exp=c_exp))
            self.network = nn.Sequential(*layers)

        self.set_device(use_cuda)

        self.name = f"PDResNet{L}"

        self.arguments = dict(
            L=L,
            Vker_size=Vker_size,
            Wker_size=Wker_size,
            c_exp=c_exp,
            use_same_block=use_same_block,
            use_cuda=use_cuda,
        )
        self.minfo = rcp.TabloidInfo(L, Vker_size, c_exp, use_same_block)

    def forward(self, x, f=None):
        if self.device != x.device:
            x = x.to(self.device)
        if f is None:
            f = x
        else:
            f = f.to(self.device)

        out, _ = self.network((x, f))
        return out
