# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:17:18 2022

@author: anton

Combines the different approaches on how to introduce various neural network approaches
intro the Chambolle-Pock algorithm.
"""

import os

from .tikhonov_extended_advanced import PDRChain
from .learnable_cp import LCP
from .learnable_lp_residual import ComboLCPPDResNet
from .recurrent import RecurrentNet
from .residual import ResidualNet
from .residual_advanced import APDNet
from .tikhonov_residual import Pdresnet
from .residual_extended import ResNetChain

modelmapping = {
    'EPDN': ResNetChain,
    'LCP': LCP,
    'LPDN': ComboLCPPDResNet,
    'PDSM': RecurrentNet,
    'PDResNet': ResidualNet,
    'APDN': Pdresnet,
    'PDResNet_': APDNet,
    'EAPDN': PDRChain,
}

def get_model(modelname, task = 'denoising'):
    """Loads a pretrained model by the model name.

    Args:
        modelname (str): The name of the model. The corresponding files can
            be found in the trained_networks dir.
        task (str): Which task the model should do. Available are
            'denoising' and 'deblurring'.

    Returns:
        A loaded pretrained model.
    """
    path = os.path.join('models', 'trained_networks', task, f'{modelname}.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{modelname}' not found.")
    model = modelmapping[modelname].parse_model(path)
    model.name = modelname
    return model

def get_all_models(task = 'denoising'):
    """Returns all existing models.

    Args:
        task (str, optional): The task this model should do.
            Available are 'denoising' and 'deblurring'. Defaults to 'denoising'.

    Returns:
        A dictionary with all the models. The keys are the modelnames.
    """
    modeldc = {}
    for mod in modelmapping:
        print(f"Loading {mod}: ", end='') # in parse_model a print statement is done.
        modeldc[mod] = get_model(mod, task)
    return modeldc
