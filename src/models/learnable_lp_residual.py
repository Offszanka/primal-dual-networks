# -*- coding: utf-8 -*-
"""
@author: Anton
Implements LPDN. The combination of LCP and PDResNet'.
"""

import torch

from .reschapo import PDNet

from .residual_advanced import APDNet
from .learnable_cp import LCP


class ComboLCPPDResNet(PDNet):
    """A combined network of LCP and PDResNet.

    First the LCP network is used. Afterwards the result is sent through a residual network.

    Args:
            lcp (LCP.LCP): _description_
            pdresnet (PDResNet.APDNet): _description_
            use_cuda (bool, optional): If True sets the device on the cuda device by using set_device of rcp.PDNet. Defaults to True.
    """

    def __init__(self, lcp: LCP, pdresnet: APDNet, use_cuda: bool = True) -> None:
        super().__init__()
        self.lcp = lcp
        self.pdresnet = pdresnet

        self.name = "LCP-PDResNet Combo"
        self.set_device(use_cuda)
        self.arguments = dict(
            lcp_arguments=lcp.arguments,
            pdresnet_arguments=pdresnet.arguments,
        )

    @classmethod
    def parse_model(cls, path, use_cuda=True):
        tdic = torch.load(path, 'cpu')
        if tdic.get('arguments') is None:
            raise ValueError(
                f"{path} did not save the arguments. I am helpless. :(")

        pdr = APDNet(**tdic['arguments']['pdresnet_arguments'])
        lcp = LCP(**tdic['arguments']['lcp_arguments'])
        instance = cls(lcp=lcp, pdresnet=pdr, use_cuda=use_cuda)
        instance._load(path)
        return instance

    def forward(self, im):
        if im.device != self.device:
            im = im.to(self.device)
        out = self.lcp(im)
        out = self.pdresnet(out)
        return out
