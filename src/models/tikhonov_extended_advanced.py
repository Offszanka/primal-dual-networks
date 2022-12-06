# -*- coding: utf-8 -*-
"""
@author: Anton
Implements the extended APDN, EAPDN.
This approach is based on the Chambolle-Pock iteration which arises if we use it on the
Tikhonov regularization functional:
||Ax-g||_2 + lam*||nabla x||_1
"""

import math

import torch
from torch import nn

import models.reschapo as rcp

class NormalizationBlock(nn.Module):
    """Implements a normalization block used in the network."""
    def __init__(self, kernelsize, channelsize) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, num_channels=channelsize)
        self.norm2 = nn.GroupNorm(1, num_channels=channelsize)
        self.conv1 = nn.Conv2d(
            in_channels=channelsize, out_channels=channelsize,
            kernel_size=kernelsize, padding=math.floor(kernelsize/2), bias=True,
            )
        self.conv2 = nn.Conv2d(
            in_channels=channelsize, out_channels=2*channelsize,
            kernel_size=kernelsize, padding=math.floor(kernelsize/2), bias=True,
        )
        self.proxF1 = rcp.Prox_l2T(sigma=1)
        self.proxF2 = rcp.Prox_l1(sigma=1)
        self.norm3 = nn.GroupNorm(1, num_channels=channelsize)
        self.norm4 = nn.GroupNorm(1, num_channels=2*channelsize)
        self.conv3 = nn.Conv2d(
            in_channels=channelsize, out_channels=channelsize,
            kernel_size=kernelsize, padding=math.floor(kernelsize/2), bias=False,
        )
        self.conv4 = nn.Conv2d(
            in_channels=2*channelsize, out_channels=channelsize,
            kernel_size=kernelsize, padding=math.floor(kernelsize/2), bias=False,
        )
        self.alpha = nn.Parameter(torch.zeros(
            (1, channelsize, 1, 1)), requires_grad=True)

    def forward(self, inp):
        xk, f = inp
        out1 = self.norm1(xk)
        out1 = self.conv1(out1)
        out1 = self.proxF1(out1, f)
        out1 = self.norm3(out1)
        out1 = self.conv3(out1)

        out2 = self.norm2(xk)
        out2 = self.conv2(out2)
        out2 = self.proxF2(out2)
        out2 = self.norm4(out2)
        out2 = self.conv4(out2)

        out = out1 + out2
        out = xk + self.alpha*out
        return out, f

class PDRChain(rcp.PDNet):
    """Implements the extended APDN, EAPDN.
    This is based on the Chambolle-Pock iteration which arises if we use it on the
    Tikhonov regularization functional:
    ||Ax-g||_2 + lam*||nabla x||_1

    Args:
            width (int, optional): The channel size of the network. Defaults to 16.
            blk_num (int, optional): The number of layers. Defaults to 15.
            img_channel (int, optional): The number of color channels of the input images.
                Defaults to 1.
            use_cuda (bool, optional): Sets the device to cuda if True. Uses the inherit function set_device. Defaults to True.
    """
    def __init__(self, width: int=16, blk_num: int=15,
                 img_channel: int=1, use_cuda: bool=True) -> None:
        super().__init__()
        chan = width
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                      bias=True)
        self.res = nn.Parameter(torch.zeros(
            (1, img_channel, 1, 1)), requires_grad=True)

        self.middle_blks = \
            nn.Sequential(
                *[NormalizationBlock(kernelsize=3, channelsize=chan) for _ in range(blk_num)]
            )

        self.set_device(use_cuda)
        self.name = "ResPDNafNet"

        self.arguments = dict(
            width=width,
            blk_num=blk_num,
            img_channel=img_channel,
            use_cuda=use_cuda
        )

    def forward(self, inp):
        B, C, H, W = inp.shape
        x = self.intro(inp)
        f = x.clone().detach()
        x, f = self.middle_blks((x, f))
        x = self.ending(x)
        x = self.res*x + inp

        return x[:, :, :H, :W]