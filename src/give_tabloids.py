# -*- coding: utf-8 -*-
"""
Created on Sat Sep 2 03:25:20 2022

@author: anton

A script to compare several models and display their results in a table.

After restructering the whole structure of this code this may not work anymore.
"""

import os
from collections import OrderedDict, namedtuple
from os.path import join

import dset
import models
import numpy as np
import pandas as pd
import tqdm
from dset import r
from misc import psnr

comparing2500 = False
comparing10k = True

_, test_set = dset.getDefaultSet(transform=dset.NoCrop(stddev=25/255),
                                 max_test_size=200, max_train_size=1, preload=True)
if comparing2500:
    # Compare the models after the short 2500 iteration runs.

    fp = "comparison_networks"  # The path leading to the trained models.
    n2m = {
        "PDResNet": models.ResidualNet,
        "PDResNet'": models.APDNet,
        "PDSM": models.RecurrentNet,
        "LCP": models.LCP,
    }

    lists_base = {"PDResNet": join(fp, 'pdresnet'), "PDResNet'": join(
        fp, 'pdresnet_'), "PDSM": join(fp, 'pdsm')}
    modelnames = {**lists_base,
                  **dict(LCP=join(fp, 'lcp'))
                  }

    dtab = dict()
    dtab['model_name'] = []
    dtab['channelsize'] = []
    dtab['kernelsize'] = []
    dtab['shared'] = []
    dtab['psnr'] = []
    dtab['layer'] = []
    for k in modelnames:
        sdirs = os.listdir(modelnames[k])
        for r, d, f in os.walk(modelnames[k]):
            for file in f:
                t, ext = os.path.splitext(file)
                if ext == '.pth':
                    path = join(r, file)
                    print(f"Working on {path}")
                    model = n2m[k].parse_model(path)
                    tdc = model.validate(test_set, batch_size=50)
                    dtab['channelsize'].append(model.minfo.channelsize)
                    dtab['kernelsize'].append(model.minfo.kernelsize)
                    dtab['shared'].append(model.minfo.shared)
                    dtab['layer'].append(model.minfo.layer)
                    dtab['psnr'].append(round(tdc['psnr'], 2))
                    dtab['model_name'].append(k)
    tabloid = pd.DataFrame(data=dtab)
    tabloid = tabloid.fillna(0)
    ptab = tabloid.pivot_table(values='psnr',
                               index=['shared', 'model_name',
                                      'kernelsize', 'channelsize', 'layer'],
                               aggfunc=np.max)
    ptab.to_html('comparison_2500.html')

if comparing10k:
    # Comparing the final runs.
    import PyChaPo as pcp
    from skimage.metrics import structural_similarity as ssim

    m2n = {'combo': models.ComboLCPPDResNet,
           'LCP': models.LCP,
           'PDResNet_': models.APDNet,
           'PDResNetChain': models.ResNetChain,
           'PDResNet': models.ResidualNet,
           'PDSM': models.RecurrentNet,
           }

    dtab = dict()
    dtab['model_name'] = []
    dtab['psnr'] = []
    dtab['ssim'] = []

    print("Mean psnr = ", psnr(
        test_set[:]['clean'], test_set[:]['noisy']).mean())

    m2n = {
        'LPDN': models.ComboLCPPDResNet,
        'LCP': models.LCP,
        'PDResNet_': models.APDNet,
        'PDResNet': models.ResidualNet,
        'PDSM': models.RecurrentNet,
        'APDN': models.Pdresnet,
        'EAPDN': models.ResNetChain,
        'DCT3': pcp.LPychapo,
        'DCT5': pcp.LPychapo,
        'fst_plus_snd': pcp.LPychapo,
    }

    models = OrderedDict()

    def get_model(modelname):
        path = os.path.join('models', 'trained_networks', 'denoising', f'{modelname}.pth')
        model = m2n[modelname].parse_model(path)
        model.name = modelname
        models[modelname] = model
        return model

    for mod in m2n:
        get_model(mod)

    def convert(im):
        im = np.asarray(im)
        if len(im.shape) == 3:
            im = im[0]
        if im.dtype == np.float64:
            im = im.astype(np.float32)
        return im

    sls = np.zeros(len(test_set))
    for i, im in enumerate(test_set):
        cim, nim = dset.r(im)
        sls[i] = ssim(convert(cim), convert(nim))

    print("Mean SSIM:", sls.mean())

    def val(set, model):
        psnrls = np.zeros(len(set))
        ssimls = np.zeros(len(set))
        for i, im in enumerate(set):
            cim, nim = dset.r(im)
            dim = model.denoise(nim)
            psnrls[i] = psnr(dim, cim)
            ssimls[i] = ssim(convert(dim), convert(cim))
        return psnrls, ssimls

    for k in tqdm.tqdm(models):
        mod = models[k]
        psnrls, ssimls = val(test_set, mod)
        dtab['model_name'].append(k)
        dtab['psnr'].append(round(psnrls.mean(), 2))
        dtab['ssim'].append(round(ssimls.mean()*100, 2))

    tabloid = pd.DataFrame(data=dtab)
    tabloid = tabloid.T
    tabloid = tabloid.reset_index(drop=True)
    # print(tabloid.columns.to_list())
    # tabloid = tabloid[[10, 1, 7, 6, 3, 8, 9, 4, 0, 5, 2]]
    print(tabloid)
    tabloid = tabloid.fillna(0)
    ptab_psnr = tabloid.pivot_table(values='psnr',
                                    index=['shared', 'model_name',
                                           'kernelsize', 'channelsize', 'layer'],
                                    aggfunc=np.max)
    ptab_ssim = tabloid.pivot_table(values='psnr',
                                    index=['model_name',
                                           'kernelsize', 'channelsize', 'layer'],
                                    aggfunc=np.max)
    tabloid.to_html('comparison_10k.html')
    tabloid.to_latex('comparison_10k.tex')
