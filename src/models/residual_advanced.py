# -*- coding: utf-8 -*-
"""
Implements the primal-dual residual network again.
However, this time we use some adjustments to boost the performance of the network (PSNR wise).

In short: PDResNet'
"""

import math

import torch
import torch.nn as nn

import models.reschapo as rcp


class NormalizationBlock(nn.Module):
    """The Block with the adjustments
    """

    def __init__(self, W_kernel_size=3, V_kernel_size=3, c_exp=2, inc=1, outc=1) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(1, num_channels=inc)
        self.conv1 = nn.Conv2d(
            in_channels=inc, out_channels=c_exp,
            kernel_size=W_kernel_size, padding=math.floor(W_kernel_size/2), bias=True,
        )
        self.prox_F = rcp.Prox_l1()
        self.norm2 = nn.GroupNorm(1, num_channels=c_exp)
        self.conv2 = nn.Conv2d(
            in_channels=c_exp, out_channels=outc,
            kernel_size=V_kernel_size, padding=math.floor(V_kernel_size/2), bias=False,
        )
        self.prox_G = rcp.Prox_l2_learnable(tau=1e-5)
        self.alpha = nn.Parameter(torch.zeros(
            (1, inc, 1, 1)), requires_grad=True)

    def forward(self, input):
        xk, f = input
        out = self.norm1(xk)
        out = self.conv1(out)
        out = self.prox_F(out)

        out = self.norm2(out)
        out = self.conv2(out)
        # NOTE: Hier wurde geändert. Bestehende Netze könnten unzufriedend sein.
        out = xk + self.alpha * out
        out = self.prox_G(out, f)

        return out, f


class APDNet(rcp.PDNet):
    """Implements PDResNet' which is an improved version of PDResNet.
    It has better"""

    def __init__(self, L: int = 1,
                 Vker_size: int = 3, Wker_size: int = 3, c_exp: int = 1,
                 use_same_block: bool = False,
                 use_cuda: bool = True,
                 ):
        """
        Args:
            L (int, optional): Number of layers. Defaults to 1.
            Vker_size (int, optional): The kernel size of the V matrix. Defaults to 3.
            Wker_size (int, optional): The kernel size of the V matrix. Defaults to 3.
            c_exp (int, optional): Expanse factor of the channel size. Defaults to 1.
            use_same_block (bool, optional): Determines if the parameters should be shared. Defaults to False.
            use_cuda (bool, optional): If True sets the device on the cuda device by using set_device of rcp.PDNet. Defaults to True.
        """
        super().__init__()
        self.L = L
        self.c_exp = c_exp

        if use_same_block:
            self.block = NormalizationBlock(W_kernel_size=Wker_size,
                                            V_kernel_size=Vker_size, c_exp=c_exp)
            self.network = nn.Sequential(*(L*[self.block]))
        else:
            layers = []
            for _ in range(L):
                layers.append(NormalizationBlock(W_kernel_size=Wker_size,
                                                 V_kernel_size=Vker_size, c_exp=c_exp))
            self.network = nn.Sequential(*layers)

        self.set_device(use_cuda)
        self.name = "PDResNet"
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
