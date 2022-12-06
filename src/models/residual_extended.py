# -*- coding: utf-8 -*-
"""
@author: Anton
Implements the EPDN. The extended PDResNet' with a pre and post processor.
"""

import math

import torch
import torch.nn as nn

import models.reschapo as rcp


class NormalizationBlock(nn.Module):
    """Implements a block of the ResNetChain"""
    def __init__(self, inp, W_kernel_size=3, V_kernel_size=3, lam=1/0.044) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(1, num_channels=inp)
        self.conv1 = nn.Conv2d(
            in_channels=inp, out_channels=2*inp,
            kernel_size=W_kernel_size, padding=math.floor(W_kernel_size/2), bias=True,
            groups=inp)
        self.prox_F = rcp.Prox_l1()
        self.norm2 = nn.GroupNorm(2*inp, num_channels=2*inp)
        self.conv2 = nn.Conv2d(
            in_channels=2*inp, out_channels=inp,
            kernel_size=V_kernel_size, padding=math.floor(V_kernel_size/2), bias=False,
        )
        self.prox_G = rcp.Prox_l2_learnable(tau=lam*0.1)
        self.res = nn.Parameter(torch.zeros(
            (1, inp, 1, 1)), requires_grad=True)

    def forward(self, input):
        xk, f = input
        out = self.norm1(xk)
        out = self.conv1(out)
        out = self.prox_F(out)

        out = self.norm2(out)
        out = self.conv2(out)
        out = xk + out*self.res
        out = self.prox_G(out, f)

        return out, f

class ResNetChain(rcp.PDNet):
    """Implements the EPDN. The extended PDResNet' with a pre and post processor."""

    def __init__(self, width: int = 16, middle_blk_num: int = 15,
                 img_channel: int = 1, use_cuda: bool = True) -> None:
        """
        Args:
            width (int, optional): Channelsize of the inner blocks. Defaults to 16.
            middle_blk_num (int, optional): _description_. Defaults to 15.
            img_channel (int, optional): _description_. Defaults to 1.
            use_cuda (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        chan = width
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                      bias=True),
            # nn.GELU(),
            # nn.Conv2d(in_channels=img_channel, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
            #           bias=True),
        )
        # self.ending = OutroBlock(width, img_channel)
        self.middle_blks = nn.ModuleList()
        self.res = nn.Parameter(torch.zeros(
            (1, img_channel, 1, 1)), requires_grad=True)

        self.middle_blks = \
            nn.Sequential(
                *[NormalizationBlock(chan) for _ in range(middle_blk_num)]
            )

        self.set_device(use_cuda)
        self.name = "ResPDNafNet"

        self.arguments = dict(
            width=width,
            middle_blk_num=middle_blk_num,
            img_channel=img_channel,
            use_cuda=use_cuda
        )

    def forward(self, inp):
        if inp.device != self.device:
            inp = inp.to(self.device)
        B, C, H, W = inp.shape
        x = self.intro(inp)
        f = x.clone().detach()
        x, f = self.middle_blks((x, f))
        x = self.ending(x)
        x = self.res*x + inp

        return x[:, :, :H, :W]
