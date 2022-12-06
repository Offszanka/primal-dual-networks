# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:49:58 2022

@author: Anton
"""

import argparse
import math
import os
from datetime import datetime
import re
from typing import Callable, OrderedDict, Tuple
from collections import OrderedDict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import dset
import PyChaPo as pych
from PyChaPo import _gradient, disDivergence
import reschapo as rcp
from misc import (add_gaussian_noise, gauss_blurring, load_network,
                  plot_history, psnr, show, showpsnr)

SAVE_MODELS = True
if not SAVE_MODELS:
    print("VORSICHT, die Modelle in den kurzen Läufen werden nicht gespeichert")


def save_model(model, name):
    model.save_model(os.path.join('comparison_networks', f"{name}.pth"))


def print_results(text, inbetween, dc, destinction_name, dir_name=""):
    print(text)
    for k in dc:
        print(f"{inbetween}{k} - Validating: {dc[k].psnr:.2f}dB")
        if SAVE_MODELS:
            path=os.path.join('comparison_networks', dir_name, f"{destinction_name}_{k}.pth")
            dc[k].model.save_model(path)


def train_run(model, layer, train_size=200, test_size=200, batch_size=32, epochs=2500, tqdm_description="Training",
              run_idx=None, description="",
              train_transform=None, test_transform=None):
    stddev = 25/255
    if train_transform is None: 
        train_transform = dset.StandardTransform(stddev=stddev, rcropsize=256)
    if test_transform is None: 
        test_transform = dset.NoCrop(stddev=stddev)
        
    _, test_set = dset.getDefaultSet(transform=test_transform, 
                                     max_test_size=200, max_train_size=1, preload=True)
    if run_idx is None:
        final_save_path = None
        save_checkpoint_path = None
        checkpoint_interval = 0
        load_checkpoint_path = None
    else:
        final_save_path = 'auto'
        save_checkpoint_path = os.path.join("curious_networks", 'data')
        checkpoint_interval = 50
        if os.path.exists(os.path.join('curious_networks', 'data', run_idx)):
            print(
                f"Found existing checkpoint to corresponding {run_idx=}\nWill resume training.")
            load_checkpoint_path = os.path.join(
                'curious_networks', 'data', run_idx)
        else:
            load_checkpoint_path = None

    fit_args = dict(batch_size=batch_size, epochs=epochs,
                    optimizer_args={'lr': 1e-3, 'betas': (0.9, 0.9)},
                    scheduler_args={
                        'T_max': epochs, 'eta_min': 1e-6},
                    final_save_path=final_save_path,
                    run_idx=run_idx,
                    save_checkpoint_path=save_checkpoint_path,
                    save_checkpoint_interval=checkpoint_interval,
                    tqdm_description=tqdm_description,
                    test_set_validate_interval=50,
                    load_checkpoint_path=load_checkpoint_path,
                    description=description)
    hist, _ = model.conv_fit(train_size=train_size, test_size=test_size,
                            test_transform=test_transform,
                            train_transform=train_transform,
                             fit_args=fit_args)

    tdc = model.validate(test_set, batch_size=25)
    print(f"\n\t{layer}-Validate: {tdc['psnr']:.2f}dB\n")
    return tdc, model


def make_image(test_set, name, model):
    nim = test_set[25]['noisy']
    fname = os.path.join('ex_bilder', f"{name}.png")
    torchvision.utils.save_image(model.denoise(nim), fname)


if __name__ == '__main__':
    import script5 as PDSM
    import script6 as PDResNet
    import script7 as nafnet
    import script8 as PDResNet_
    import script9 as LCP
    import script10 as Combos
    import script11 as pdresnetnafnet
    import script12 as resnetchain
    import script13 as deblur
    import script14 as eapdn

    MRes = namedtuple("mresult", "model psnr")
    _, test_set = dset.getDefaultSet(transform=dset.NoCrop(stddev=25/255), max_test_size=200, max_train_size=1, preload=True)
    # Switch between True and False
    mode = "full_training" if True else "testing"
    print(f"{mode=}")
    if mode == "testing":
        epochs = 2500

        train_pdsm = not True
        train_pdresnet = not True
        train_pdresnet_ = not True
        train_PDResNetChain = not True

        train_pdresnet_layers = not True
        train_pdsm_layers = not True
        train_pdresnet_STRICH_layers = not True
        train_lcp = not True
        
        

        # layerlist_lcp = [45]
        # lcp_m = OrderedDict()
        # lcp = LCP.LCPOutro(use_cuda=True, max_iter=45, lam=1/0.044, use_same_block=False)
        # tdc, _ = train_run(lcp, 45, tqdm_description=f"LCPOutro2-{45}",
        #                     epochs=epochs)
        # print_results(f"Results kernelsizes {True=}", "ksize",
        #                       {45 : MRes(lcp, tdc['psnr'])}, f"kernelsize_shared_outro")
        
        # lcp = LCP.LCPOutro2(use_cuda=True, max_iter=45, lam=1/0.044, use_same_block=False,
        #                     pretrain_path=os.path.join('ready_trained', 'lcp.pth'))
        # tdc, _ = train_run(lcp, 45, tqdm_description=f"LCPOutro2-{45}",
        #                     epochs=epochs)
        # print_results(f"Results kernelsizes {True=}", "ksize",
        #                       {45 : MRes(lcp, tdc['psnr'])}, f"kernelsize_shared_outro2")
        
        
        # pdresnetpp = PDResNet_.PDResNetPP(L=45, use_cuda=True,
        #                    Vker_size=3, Wker_size=3, c_exp=8,
        #                    use_same_block=False)
        # tdc,_ = train_run(pdresnetpp, 45, tqdm_description="LCP-PDResNet-2",
        #           epochs=epochs)
        # print_results(f"Results kernelsizes {True=}", "ksize",
        #                 {45 : MRes(pdresnetpp, tdc['psnr'])}, f"kernelsize_sunhared_pdresnetpp")

        # Training the combination of LCP and PDResNet
        # lcp = LCP.LCP(use_cuda=True, max_iter=45, lam=1/0.044, use_same_block=False)
        # pdresnet = PDResNet_.APDNet(L=45, use_cuda=True,
        #                 Vker_size=7, Wker_size=7, c_exp=8,
        #                 description="", use_same_block=False)
        # lcp_pdresnet = Combos.ComboLCPPDResNet(lcp, pdresnet, use_cuda=True)
        # train_run(lcp, "Erster Run LCP-PDResNet", tqdm_description="(LCP-PDResNet)-1",
        #         epochs=500)
        # train_run(pdresnet, 45, tqdm_description="(LCP-PDResNet)-2",
        #         epochs=500)
        # tdc, _ = train_run(lcp_pdresnet, 45, tqdm_description="(LCP-PDResNet)-3",
        #                 epochs=epochs, run_idx = "combo-run",
        #                 description="Der finale Combo-Run mit LCP-PDResNet'. Beide Layer-Anzahl ist 45.")
        # print(f"LCP-PDResNet Validate: {tdc['psnr']:.2f}dB")
        # make_image(test_set, "LCPPDResNet Combo", lcp_pdresnet)

        # Training the PDResNet

        # use_same_block = True
        # batch_size = 32
        # train_size = 200
        # test_size = 200
        # layerlist = [1,2,4,8,16]
        # for layer in layerlist:
        #     pdresnet = PDResNet.ResidualNet(L=15, use_cuda=True,
        #                         Vker_size=3, Wker_size=3, c_exp=layer,
        #                         description="", use_same_block=use_same_block)
        #     tdc_pdr, _ = train_run(pdresnet, layer, tqdm_description=f"PDResNet-{layer}")

        #     pdr_ = PDResNet_.APDNet(L=15, use_cuda=True,
        #                             Vker_size=3, Wker_size=3, c_exp=layer,
        #                             description="", use_same_block=use_same_block,
        #                             bArch=PDResNet_.NormalizationBlock)
        #     tdc_pdr_, _ = train_run(pdr_, layer, tqdm_description=f"PDResNet_-{layer}")

        #     pdsm = PDSM.RecurrentNet(L=layer, use_cuda=True,
        #                             Vker_size=3, Wker_size=3, y_size=16,
        #                             description="", use_same_block=use_same_block)
        #     tdc_pdsm, _ = train_run(pdresnet, layer, tqdm_description=f"PDSM-{layer}")

        if train_pdresnet_:
            for shared_parameters in [False, True]:
                kernelsizes = [3, 5, 7]
                channelsizes = [1, 2, 4, 8, 16]
                mdic1 = dict()
                mdic2 = dict()
                for ksize in kernelsizes:
                    pdresnet = PDResNet_.APDNet(L=15, use_cuda=True,
                                                Vker_size=ksize, Wker_size=ksize, c_exp=1,
                                                # description=f"layer=15, kernelsize={ksize}, c_exp=1",
                                                use_same_block=shared_parameters)
                    tdc, _ = train_run(pdresnet, f"kernel {ksize}x{ksize}", tqdm_description=f"pdresnet_-kernelsize{ksize}",
                                       epochs=epochs)
                    mdic1[ksize] = MRes(pdresnet, tdc['psnr'])

                for css in channelsizes:
                    pdresnet = PDResNet_.APDNet(L=15, use_cuda=True,
                                                Vker_size=3, Wker_size=3, c_exp=css,
                                                # description=f"layer=15, kernelsize=3, c_exp={css}",
                                                use_same_block=shared_parameters)
                    tdc, _ = train_run(pdresnet, f"channel {css}", tqdm_description=f"pdresnet_-channelsize{css}",
                                       epochs=epochs)
                    mdic2[css] = MRes(pdresnet, tdc['psnr'])
                print_results(f"Results kernelsizes {shared_parameters=}", "ksize",
                              mdic1, f"kernelsize_shared{shared_parameters}", 'pdresnet_')
                print_results(f"Results channelsize {shared_parameters=}", "csize",
                              mdic2, f"channelsize_shared{shared_parameters}", 'pdresnet_')

        if train_pdresnet:
            for shared_parameters in [True, False]:
                kernelsizes = [3, 5, 7]
                channelsizes = [1, 2, 4, 8, 16]
                mdic1 = dict()
                mdic2 = dict()
                for ksize in kernelsizes:
                    pdresnet = PDResNet.ResidualNet(L=15, use_cuda=True,
                                                    Vker_size=ksize, Wker_size=ksize, c_exp=1,
                                                    # description=f"layer=15, kernelsize={ksize}, c_exp=1",
                                                    use_same_block=shared_parameters)
                    tdc, _ = train_run(pdresnet, f"kernel {ksize}x{ksize}", tqdm_description=f"pdresnet-kernelsize{ksize}",
                                       epochs=epochs)
                    mdic1[ksize] = MRes(pdresnet, tdc['psnr'])

                for css in channelsizes:
                    pdresnet = PDResNet.ResidualNet(L=15, use_cuda=True,
                                                    Vker_size=3, Wker_size=3, c_exp=css,
                                                    # description=f"layer=15, kernelsize=3, c_exp={css}",
                                                    use_same_block=shared_parameters)
                    tdc, _ = train_run(pdresnet, f"channel {css}", tqdm_description=f"pdresnet-channelsize{css}",
                                       epochs=epochs)
                    mdic2[css] = MRes(pdresnet, tdc['psnr'])
                print_results(f"Results kernelsizes {shared_parameters=}", "ksize",
                              mdic1, f"kernelsize_shared{shared_parameters}", 'pdresnet')
                print_results(f"Results channelsize {shared_parameters=}", "csize",
                              mdic2, f"channelsize_shared{shared_parameters}", 'pdresnet')

        # Training PDSM
        if train_pdsm:
            for shared_parameters in [True, False]:
                kernelsizes = [3,5,7]
                channelsizes = [1,2,4,8,16]
                mdic1 = dict()
                mdic2 = dict()
                for ksize in kernelsizes:
                    pdsm = PDSM.RecurrentNet(L=15, use_cuda=True,
                                             Vker_size=ksize, Wker_size=ksize, y_size=1,
                                            #  description=f"layer=15, kernelsize={ksize}, y_size=1",
                                             use_same_block=shared_parameters)
                    tdc, _ = train_run(pdsm, f"kernel {ksize}x{ksize}", tqdm_description=f"PDSM-kernelsize{ksize}",
                                       epochs=epochs)
                    mdic1[ksize] = MRes(pdsm, tdc['psnr'])

                for css in channelsizes:
                    pdsm = PDSM.RecurrentNet(L=15, use_cuda=True,
                                             Vker_size=3, Wker_size=3, y_size=css,
                                            #  description=f"layer=15, kernelsize=3, y_size={css}",
                                             use_same_block=shared_parameters)
                    tdc, _ = train_run(pdsm, f"channel {css}", tqdm_description=f"PDSM-channelsize{css}",
                                       epochs=epochs)
                    mdic2[css] = MRes(pdsm, tdc['psnr'])
                print_results(f"Results kernelsizes {shared_parameters=}", "ksize",
                              mdic1, f"kernelsize_shared{shared_parameters}", 'pdsm')
                print_results(f"Results channelsize {shared_parameters=}", "csize",
                              mdic2, f"channelsize_shared{shared_parameters}", 'pdsm')

        if train_pdresnet_layers:
            for shared_parameters in [True, False]:
                layerlist = [5, 10, 15, 30, 45]
                mdic1 = dict()
                y_size = 4 if shared_parameters else 16
                ksize = 3 if shared_parameters else 7
                for layer in layerlist:
                    pdresnet = PDResNet.ResidualNet(L=layer, use_cuda=True,
                                                    Vker_size=ksize, Wker_size=ksize, c_exp=y_size,
                                                    # description=f"layer={layer}, kernelsize={ksize}, c_exp={y_size}",
                                                    use_same_block=shared_parameters)
                    tdc, _ = train_run(pdresnet, f"layer {layer}, shared{shared_parameters}", tqdm_description=f"pdresnet-L{layer}K{ksize}C{y_size}",
                                       epochs=epochs)
                    mdic1[layer] = MRes(pdresnet, tdc['psnr'])
                print_results(f"Results PDResNet layer {shared_parameters=}", f"layer {shared_parameters}",
                              mdic1, f"layer_shared{shared_parameters}", os.path.join('pdresnet', 'layer'))

        if train_pdsm_layers:
            for shared_parameters in [True, False]:
                layerlist = [5, 10, 15, 30, 45]
                mdic1 = dict()
                y_size = 1 if shared_parameters else 16
                ksize = 5 if shared_parameters else 3
                for layer in layerlist:
                    pdsm = PDSM.RecurrentNet(L=layer, use_cuda=True,
                                             Vker_size=ksize, Wker_size=ksize, y_size=y_size,
                                            #  description=f"layer={layer}, kernelsize={ksize}, y_size={y_size}",
                                             use_same_block=shared_parameters)
                    tdc, _ = train_run(pdsm, f"layer {layer}, shared{shared_parameters}", tqdm_description=f"PDSM-L{layer}K{ksize}C{y_size}",
                                       epochs=epochs)
                    mdic1[layer] = MRes(pdsm, tdc['psnr'])

                print_results(f"Results PDSM layer {shared_parameters=}", f"layer {shared_parameters}",
                              mdic1, f"layer_shared{shared_parameters}", os.path.join('pdsm', 'layer'))

        if train_pdresnet_STRICH_layers:
            for shared_parameters in [False]:
                layerlist = [30] # [5, 10, 15, 30, 45]
                mdic1 = dict()
                ksize = 5 if shared_parameters else 7
                y_size = 16
                for layer in layerlist:
                    bs = 32 if layer < 45 else 25
                    pdresnet = PDResNet_.APDNet(L=layer, use_cuda=True,
                                                Vker_size=ksize, Wker_size=ksize, c_exp=y_size,
                                                # description=f"layer={layer}, kernelsize={ksize}, c_exp={y_size}",
                                                use_same_block=shared_parameters)
                    tdc, _ = train_run(pdresnet, f"layer {layer}, shared{shared_parameters}", tqdm_description=f"pdresnet-L{layer}K{ksize}C{y_size}",
                                       epochs=epochs, batch_size=bs)
                    mdic1[layer] = MRes(pdresnet, tdc['psnr'])
                    print_results(f"Results PDRSNet' layer {shared_parameters=}", f"layer {shared_parameters}",
                                  mdic1, f"layer_shared{shared_parameters}", os.path.join('pdresnet_', 'layer'))

        # Training LCP
        if train_lcp:
            for shared_parameters in [True, False]:
                layers = [5, 10, 15, 30, 45, 80, 100, 200]
                mdic1 = dict()
                for mi in layers:
                    lcp = LCP.LCP(max_iter=mi, use_cuda=True,
                                  lam=1/0.04, use_same_block=shared_parameters)
                    tdc, _ = train_run(lcp, f"max_iter: {mi}", tqdm_description=f"LCP-layer-{mi}",
                                       epochs=epochs)
                    mdic1[mi] = MRes(lcp, tdc['psnr'])

                    print_results(f"Results layers lcp {shared_parameters=}",
                                  "max_iter", mdic1, f"layer_shared{shared_parameters}", 'lcp')

        if train_PDResNetChain:
            print("Training PDResNetChain")
            model = resnetchain.ResNetChain(middle_blk_num=15, width=16)
            tdc, _ = train_run(model, "PDResNetChain-Training Run", epochs=epochs,
                               batch_size=25, tqdm_description="PDResNetChain",
                               description="Finaler PDResNetChain-Run. Die width nummer ist 16, middle_blk_num=15. Trainiert für stddev=25/255")

            print(f"PDResNetChain Validate: {tdc['psnr']:.2f}dB")
            make_image(test_set, "PDResNetChain Final", model)
            # save_model(model, "PDResNetChain-")

    if mode == "full_training":
        epochs = 10000
        train_lcp = not True
        train_pdsm = not True
        train_pdresnet = not True
        train_pdresnet_ = not True
        train_lcppdresnet = not True
        train_nafnet = not True
        train_pdresnetnafnet = not True
        train_PDResNetChain = not True
        
        train_apdn = not True
        train_nafnet_deblurring = not True
        train_eapdn = True
        

        # Training the LCP
        if train_lcp:
            lcp = LCP.LCP(use_cuda=True, max_iter=45, lam=1 /
                          0.035, use_same_block=False)
            tdc, _ = train_run(lcp, "LCP-Training Run", epochs=epochs, batch_size=32,
                               tqdm_description="LCP",
                               run_idx="LCP-run-final",
                               description="Der finale LCP-Run. Die Layer-Anzahl ist 45. Trainiert für stddev=25/255")
            print(f"LCP Validate: {tdc['psnr']:.2f}dB")
            make_image(test_set, "LCP Pure Final", lcp)
            lcp.save_model(os.path.join('ready_trained', 'lcp.pth'))

        # Training PDSM
        if train_pdsm:
            pdsm = PDSM.RecurrentNet(L=30, use_cuda=True,
                                         Vker_size=3, Wker_size=3, y_size=16,
                                         use_same_block=False)
            tdc, _ = train_run(pdsm, "PDSM-Training Run", epochs=epochs, batch_size=32,
                               tqdm_description="PDSM",
                               run_idx="PDSM-run-final",
                               description="Der finale PDSM-Run. Die Layer-Anzahl ist 30. Trainiert für stddev=25/255")
            print(f"PDSM Validate: {tdc['psnr']:.2f}dB")
            make_image(test_set, "PDSM Pure Final", pdsm)

        if train_pdresnet:
            pdresnet = PDResNet.ResidualNet(L=30, use_cuda=True,
                                    Vker_size=7, Wker_size=7, c_exp=16,
                                    use_same_block=False)
            tdc, _ = train_run(pdresnet, "PDResNet-Training Run", epochs=epochs, batch_size=32,
                               tqdm_description="PDResNet",
                               run_idx="PDResNet-run-final",
                               description="Der finale PDResNet-Run. Die Layer-Anzahl ist 30. Trainiert für stddev=25/255")
            print(f"PDResNet Validate: {tdc['psnr']:.2f}dB")
            make_image(test_set, "PDResNet Pure Final", pdresnet)

        if train_pdresnet_:
            pdresnet_ = PDResNet_.APDNet(L=45, use_cuda=True,
                                        Vker_size=7, Wker_size=7, c_exp=16,
                                        use_same_block=False)
            tdc, _ = train_run(pdresnet_, "PDResNet_-Training Run", epochs=epochs, batch_size=25,
                               tqdm_description="PDResNet'",
                               run_idx="PDResNet_Blurring-run-final",
                               train_transform=dset.BlurCropNoise(rcropsize=256, kernel_size=5, sigma=(2,4), stddev=0.05),
                               test_transform=dset.BlurNoise(kernel_size=5, sigma=(2,4), stddev=0.05),
                               description="Der finale PDResNet'-Run. Die Layer-Anzahl ist 30. Trainiert für stddev=25/255"
                               )
            print(f"PDResNet' Validate: {tdc['psnr']:.2f}dB")
            make_image(test_set, "PDResNet_ Pure Final", pdresnet_)

        # Training the combination of LCP and PDResNet
        if train_lcppdresnet:
            print("Training LCP - PDResNet Combo")
            if os.path.exists(os.path.join('ready_trained', 'pdresnet_.pth')):
                pdresnet = PDResNet_.APDNet.parse_model(os.path.join('ready_trained', 'pdresnet_.pth'))
            else:
                pdresnet = PDResNet_.APDNet(L=45, use_cuda=True,
                                            Vker_size=7, Wker_size=7, c_exp=8,
                                            use_same_block=False)
                train_run(pdresnet, 45, tqdm_description="(LCP-PDResNet)-PDR", epochs=750)
                pdresnet.save_model(os.path.join('ready_trained', 'pdresnet_.pth'))
                
            if os.path.exists(os.path.join('ready_trained', 'lcp.pth')):
                lcp = LCP.LCP.parse_model(os.path.join('ready_trained', 'lcp.pth'))
            else:
                lcp = LCP.LCP(use_cuda=True, max_iter=45,
                          lam=1/0.053, use_same_block=False)
                train_run(lcp, 45, tqdm_description="(LCP-PDResNet)-LCP", epochs=750)
                lcp.save_model(os.path.join('ready_trained', 'pdresnet_.pth'))
            
            lcp_pdresnet = Combos.ComboLCPPDResNet(
                lcp, pdresnet, use_cuda=True)
            tdc, _ = train_run(lcp_pdresnet, 45, tqdm_description="(LCP-PDResNet)-AIO",
                               epochs=epochs, 
                               run_idx="combo-run-BLUR",
                               description="Der finale Combo-Run mit LCP-PDResNet'. Beide Layer-Anzahl ist 45.FÜr stddev=25/255 trainiert. Dieses Mal wirklich.")
            print(f"LCP-PDResNet Validate: {tdc['psnr']:.2f}dB")
            make_image(test_set, "LCPPDResNet Combo", pdresnet)

        if train_nafnet:
            print("Training NAFNet")
            nnet = nafnet.ChensNetwork(img_channel=1, width=32,
                                       middle_num=12, left_num=[2, 2, 4, 8], right_num=[2, 2, 2, 2], use_cuda=True)
            tdc, _ = train_run(nnet, "NafNet", tqdm_description="NafNet",
                               epochs=epochs, 
                               batch_size=16,
                               train_transform=dset.BlurCropNoise(rcropsize=256, kernel_size=5, sigma=(2,4), stddev=0.05),
                               test_transform=dset.BlurNoise(kernel_size=5, sigma=(2,4), stddev=0.05),
                               run_idx="NAFNet-blur-run",
                               description="""Der finale NAFNet-run. Mit width=32, middle_num=12, left_num=[2,2,4,8], right_num=[2,2,2,2]. 10 000 Epochen trainiert.
                               Außerdem trainiert für stddev=25/255.""")
            print(f"NAFNet Validate: {tdc['psnr']:.2f}dB")
            make_image(test_set, "NAFNet", nnet)

        if train_pdresnetnafnet:
            print("Training PDResNet-NAFNet")
            pupsnet = pdresnetnafnet.ResNetChain(
                enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])
            tdc, _ = train_run(pupsnet, " PDResNet-NAFNet", tqdm_description=" PDResNet-NAFNet",
                               epochs=epochs, 
                               run_idx="PDResNetIntoNAFNet-run-final",
                               description="""Der finale PDResNet-NAFNet-run. Mit width=16, middle_num=12, left_num=[2,2,4,8], right_num=[2,2,2,2]. 10 000 Epochen trainiert.
                               Außerdem trainiert für stddev=25/255 Dieses Mal wirklich""")
            print(f"PDResNetIntoNAFNet Validate: {tdc['psnr']:.2f}dB")
            make_image(test_set, "PDResNetIntoNAFNet", pupsnet)

        if train_PDResNetChain:
            print("Training PDResNetChain")
            model = resnetchain.ResNetChain(middle_blk_num=15, width=16)
            tdc, _ = train_run(model, "PDResNetChain-Training Run", epochs=epochs,
                               run_idx="PDResNetChain-BLUR", batch_size=25, tqdm_description="PDResNetChain",
                               train_transform=dset.BlurCropNoise(rcropsize=256, kernel_size=5, sigma=(2,4), stddev=0.05),
                               test_transform=dset.BlurNoise(kernel_size=5, sigma=(2,4), stddev=0.05),
                               description="Finaler PDResNetChain-Run. Die width nummer ist 16, middle_blk_num=15. Trainiert für stddev=25/255. Dieses Mal wirklich")

            print(f"PDResNetChain Validate: {tdc['psnr']:.2f}dB")
            make_image(test_set, "PDResNetChain Final", model)

        if train_apdn:
            print("Training Deblur")
            mod = deblur.Pdresnet(L = 45, kernelsize=3, channelsize=8, img_channel = 1) # The parameters for below.
            # mod = deblur.PDRChain.parse_model(os.path.join('ready_trained', 'blurnet.pth'))
            tdc, _ = train_run(mod, f"Chain", tqdm_description=f"deblur",
                                       epochs=epochs,
                                       run_idx=f"APDN-Blur",
                                       batch_size=32,
                                       train_transform=dset.BlurCropNoise(rcropsize=256, kernel_size=5, sigma=(2,4), stddev=0.05),
                                       test_transform=dset.BlurNoise(kernel_size=5, sigma=(2,4), stddev=0.05),
                                       description="""Versuch gegen Denoise mit DeblurringNetzwerk. 
                                       """
                                       )
            
        if train_eapdn:
            print("Training Deblur")
            mod = eapdn.PDRChain(width=16, blk_num=15, img_channel = 1) # The parameters for below.
            # mod = deblur.PDRChain.parse_model(os.path.join('ready_trained', 'blurnet.pth'))
            tdc, _ = train_run(mod, f"Chain", tqdm_description=f"deblur",
                                       epochs=epochs,
                                       run_idx=f"EAPDN-Blur",
                                       batch_size=25,
                                       train_transform=dset.BlurCropNoise(rcropsize=256, kernel_size=5, sigma=(2,4), stddev=0.05),
                                       test_transform=dset.BlurNoise(kernel_size=5, sigma=(2,4), stddev=0.05),
                                       description="""Versuch gegen Denoise mit DeblurringNetzwerk. 
                                       """
                                       )
            
        # if train_nafnet_deblurring:
        #     print("Training NAFNet")
        #     nnet = nafnet.ChensNetwork(img_channel=1, width=16,
        #                                middle_num=1, left_num=[1, 1, 1, 28], right_num=[1, 1, 1, 1], use_cuda=True)
        #     tdc, _ = train_run(nnet, "NafNet", tqdm_description="NafNet-Deblurring",
        #                        epochs=epochs, 
        #                        run_idx="NAFNet-blur-run",
        #                          train_transform=dset.BlurCropNoise(rcropsize=256, kernel_size=5, sigma=(1,1), stddev=0.025),
        #                         test_transform=dset.BlurNoise(kernel_size=5, sigma=(1,1), stddev=0.025),
        #                        description="""NAFNet-Versuch gegen Blurring.
        #                        BlurrCropNoise-Parameter: rcropsize=256, kernel_size=5, sigma=(1,1), stddev=0.025""")
        #     print(f"NAFNet Validate: {tdc['psnr']:.2f}dB")
        #     make_image(test_set, "NAFNet", nnet)