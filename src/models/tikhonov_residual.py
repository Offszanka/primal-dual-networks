# -*- coding: utf-8 -*-
"""
@author: Anton

Implements APDN, the Primal dual network based on the Tikhonov regularization.
"""

import math

import torch
import torch.nn as nn

import models.reschapo as rcp


class NormalizationBlock(nn.Module):
    def __init__(self, kernelsize, channelsize, img_channel=1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, num_channels=img_channel)
        self.norm2 = nn.GroupNorm(1, num_channels=img_channel)
        self.conv1 = nn.Conv2d(
            in_channels=img_channel, out_channels=channelsize,
            kernel_size=kernelsize, padding=math.floor(kernelsize/2), bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=img_channel, out_channels=channelsize,
            kernel_size=kernelsize, padding=math.floor(kernelsize/2), bias=True,
        )
        self.proxF1 = rcp.Prox_l2T(sigma=1)
        self.proxF2 = rcp.Prox_l1(sigma=1)
        self.norm3 = nn.GroupNorm(1, num_channels=channelsize)
        self.norm4 = nn.GroupNorm(1, num_channels=channelsize)
        self.conv3 = nn.Conv2d(
            in_channels=channelsize, out_channels=img_channel,
            kernel_size=kernelsize, padding=math.floor(kernelsize/2), bias=False,
        )
        self.conv4 = nn.Conv2d(
            in_channels=channelsize, out_channels=img_channel,
            kernel_size=kernelsize, padding=math.floor(kernelsize/2), bias=False,
        )
        self.alpha = nn.Parameter(torch.zeros(
            (1, img_channel, 1, 1)), requires_grad=True)

    def forward(self, input):
        xk, f = input
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


class Pdresnet(rcp.PDNet):
    """Implements APDN, the Primal dual network based on the Tikhonov regularization."""

    def __init__(self, L: int = 1,
                 kernelsize: int = 3, channelsize: int = 1, img_channel: int = 1,
                 use_same_block: bool = False,
                 use_cuda: bool = True,
                 ):
        """
        Args:
            L (int, optional): Number of layers. Defaults to 1.
            kernelsize (int, optional): The kernel size. Defaults to 3.
            channelsize (int, optional): The channel size. Defaults to 1.
            img_channel (int, optional): Number of color channels of the images. Defaults to 1.
            use_same_block (bool, optional): Determines if the parameters should be shared. Defaults to False.
            use_cuda (bool, optional): Sets the devices to cuda if True. Uses the inheritted set_device method. Defaults to True.
        """
        super().__init__()
        self.L = L
        if use_same_block:
            self.block = NormalizationBlock(kernelsize=kernelsize,
                                            channelsize=channelsize)
            self.network = nn.Sequential(*(L*[self.block]))
        else:
            layers = []
            for l in range(L):
                layers.append(NormalizationBlock(kernelsize=kernelsize,
                                                 channelsize=channelsize))
            self.network = nn.Sequential(*layers)

        self.set_device(use_cuda)
        self.name = "Blur-PDResNet"
        self.arguments = dict(
            L=L,
            kernelsize=kernelsize,
            channelsize=channelsize,
            use_same_block=use_same_block,
            use_cuda=use_cuda,
            img_channel=img_channel,
        )

    def forward(self, x, f=None):
        if self.device != x.device:
            x = x.to(self.device)
        if f is None:
            f = x
        else:
            f = f.to(self.device)
        out, _ = self.network((x, f))
        return out
