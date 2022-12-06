# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 21:10:17 2022

@author: Anton

A script to compare several models on a single image and display their results in a figure.
"""
import os
from torchvision.transforms.functional import crop

import dset
from misc import zoomshow, showdenoise

constants = {
    'Comparison 10000': True,
    'Comparison Blurring': False,
}

if constants['Comparison 10000']:
    # Compare the denoising results after running the models for 10k iterations.
    import models
    mdc = models.get_all_models('denoising')

    def cr(_im):
        """specifically designed for the ship image."""
        return crop(_im, 230, 213, 75, 75)

    im, nim = dset.get_image('schiff')

    if not os.path.isdir('vergleiche_10k'):
        os.makedirs('vergleiche_10k')

    showdenoise(2, 5, im, nim, mdc.values(),
                saves='vergleiche_10k', extra_name='ship ')
    zoomshow(2, 5, im, nim, cr, mdc.values(),
             save_file='vergleiche_10k', extension='zoom ship ')

    def cr2(_im):
        """specifically designed for the castle image."""
        return crop(_im, 80, 200, 75, 75)

    scim, snim = dset.get_image('schloss')

    showdenoise(2, 5, scim, snim, mdc.values(),
                saves='vergleiche_10k', extra_name='castle ')
    zoomshow(2, 5, scim, snim, cr2, mdc.values(),
             save_file='vergleiche_10k', extension='zoomed castle ')

if constants['Comparison Blurring']:
    # Compare the deblurring results after training the models on blurred images.
    import models
    mdc = models.get_all_models('deblurring')

    def cr(_im):
        """specifically designed for the ship image."""
        return crop(_im, 230, 213, 75, 75)

    im, nim = dset.get_image('schiff', transformation=dset.BlurNoise(
        kernel_size=5, sigma=(4, 4), stddev=0.05))

    showdenoise(2, 4, im, nim, mdc.values(),
                saves='vergleiche_blurring', extra_name='ship ')
    zoomshow(2, 4, im, nim, cr, mdc.values(),
             save_file='vergleiche_blurring', extension='zoom ship ')

    def cr2(_im):
        """specifically designed for the castle image."""
        return crop(_im, 80, 200, 75, 75)

    scim, snim = dset.get_image('schloss', transformation=dset.BlurNoise(
        kernel_size=5, sigma=(4, 4), stddev=0.05))
    showdenoise(2, 4, scim, snim, mdc.values(),
                saves='vergleiche_blurring', extra_name='castle ')
    zoomshow(2, 4, scim, snim, cr2, mdc.values(),
             save_file='vergleiche_blurring', extension='zoomed castle ')
