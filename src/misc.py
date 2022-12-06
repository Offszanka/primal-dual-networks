# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:16:27 2022

@author: Anton
"""
import collections
from collections import OrderedDict
import os
from typing import List, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import torch
import torchvision
import torch.nn as nn

import PIL
import cv2

def pimg(img, psnr, color=(1, 0, 0), zoomin=None):
    """Applies a red text on the upper right corner of the image which shows the psnr.
    The image should be a grayscale with shape (1, h, w).
    Specifically designed for the BSR image set
    Args:
        img (torch.tensor for shape (1, h, w)): im on which to apply the number
        psnr (float): the float number

    Returns:
        _type_: _description_
    """
    # if len(img.shape) == 3:
    #     img = img[0]
    if zoomin is not None:
        cropped = torchvision.transforms.functional.crop(
            img, zoomin.top, zoomin.left, zoomin.h, zoomin.w)
        nh = int(zoomin.h*zoomin.scale)
        nw = int(zoomin.w*zoomin.scale)
        resized = torchvision.transforms.functional.resize(
            cropped, (nh, nw)
        )
        img[:, 0:nh, 0:nw] = resized

    if len(img.shape) == 3:
        img = img[0]
    if img.dtype == np.float64:
        img = img.astype(np.float32)
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if zoomin is not None:
        cv2.rectangle(img, (zoomin.left, zoomin.top), (zoomin.left+zoomin.w,
                      zoomin.top+zoomin.h), zoomin.color, zoomin.thickness, cv2.FILLED)
        cv2.rectangle(img, (0, 0), (nw, nh),
                      zoomin.color, zoomin.thickness, cv2.FILLED)

    _, w, _ = img.shape
    cv2.putText(img, f"{psnr:.2f}", org=(w-80, 25), fontScale=0.85, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                color=color, thickness=2)
    # img = torchvision.transforms.functional.to_pil_image(img)
    # draw = PIL.ImageDraw.Draw(img)
    # font = PIL.ImageFont.truetype("/Library/Fonts/Arial.ttf", 16)
    # draw.text((w-50, h-20),f"{psnr:.2f}",(255),font=None, align='right')

    return img

def save_image(path, im):
    if torch.is_tensor(im):
        if len(im.shape) == 3:
            im = im[0].numpy()
        else:
            im = im.numpy()
    img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img*255)

def torch_ssim(nim, ref):
    from skimage.metrics import structural_similarity as ssim
    def convert(im):
        im = np.asarray(im)
        if len(im.shape) == 3:
            im = im[0]
        if im.dtype == np.float64:
            im = im.astype(np.float32)
        return im

    img = convert(nim)
    ref_image = convert(ref)

    return ssim(ref_image, img)*100

def psimg(img: torch.tensor, ref_image: torch.tensor, save_path: os.PathLike=None, psnr_color: Tuple[float, float, float]=(1, 1, 1),
          ssim_color: Tuple[float, float, float] = (1,1,1),
          psnr_position: Tuple[int, int]=None, ssim_position: Tuple[int, int]=None,
          pre_text_ssim="S:", pre_text_psnr="P:") -> np.ndarray:
    """
    Adds a text on the upper right corner of the image which shows the PSNR.
    Also a text below of the image which shows the SSIM.
    The image should be a grayscale with shape (h, w).

    Args:
        img (torch.tensor): im on which to apply the calculations. Should have shape (1, h, w)
        ref_image (torch.tensor): Reference image on which to calculate the PSNR/SSIM
        psnr_color (tuple(float, float, float), optional): Color of text of PSNR. Defaults to (1, 1, 1).
        ssim_color (tutuple(float, float, float), optional):  Color of text of SSIM. Defaults to (1,1,1).
        save_path (PathLike, optional): If given the image is saved at this position. Defaults to None which is no save.
        psnr_position (_type_, optional): _description_. Defaults to None.
        ssim_position (_type_, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: The modified image
    """
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as ssim

    def convert(im):
        im = np.asarray(im)
        if len(im.shape) == 3:
            im = im[0]
        if im.dtype == np.float64:
            im = im.astype(np.float32)
        return im

    img = convert(img)
    ref_image = convert(ref_image)

    tpsnr = _psnr(ref_image, img)
    tssim = ssim(ref_image, img)*100

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w, _ = img.shape
    if psnr_position is None:
        psnr_position= (w-105, 23)
    if ssim_position is None:
        ssim_position = (w-105, 50)
    # cv2.putText(img, f"{tpsnr:.2f}", org=(w-80, 23), fontScale=0.85, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             color=psnr_color, thickness=2)
    # cv2.putText(img, f"{tssim:.2f}", org=(w-80, h-5), fontScale=0.85, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             color=ssim_color, thickness=2)
    cv2.putText(img, f"{pre_text_psnr}{tpsnr:.2f}", org=psnr_position, fontScale=0.85, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                color=psnr_color, thickness=2)
    cv2.putText(img, f"{pre_text_ssim}{tssim:.2f}", org=ssim_position, fontScale=0.85, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                color=ssim_color, thickness=2)
    if save_path is not None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img*255)
    return img

def show(rows, cols, imgs, save_file_path=None, title=None, tight_layout=False, allow_subtitles = True,
         save_single=False, fig_creation={}):
    """If rows==cols==1 then you need to pass only one image."""
    if (rows == cols == 1) and not isinstance(imgs, collections.abc.Mapping):
        imgs = [imgs]

    # Make a dict if it is not one already.
    if not isinstance(imgs, collections.abc.Mapping):
        imgs = {i: img for i, img in enumerate(imgs)}

    if len(imgs) > cols*rows:
        warnings.warn(
            "imgs has more images then rows*cols={rows*cols}. Will truncate the list of images after reaching the number of images.")

    fig, axs = plt.subplots(ncols=cols, nrows=rows, squeeze=False, **fig_creation,
                            )
    fig.suptitle(title)
    for i, title in enumerate(imgs):
        if i >= cols*rows:
            return
        img = imgs[title]
        ax = axs.ravel()[i]

        if torch.is_tensor(img):
            img = img.to('cpu')
            img = img.detach()
            img = torchvision.transforms.functional.to_pil_image(img)

        ax.imshow(np.asarray(img), cmap=plt.gray())
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if allow_subtitles:
            ax.set_title(title)
    if save_file_path is not None:
        print("Saving the plot to ", save_file_path)
        fig.savefig(save_file_path)
        if save_single:
            for title in imgs:
                t, ext = os.path.splitext(save_file_path)
                save_image(f"{t}_{title}{ext}", imgs[title])
    if tight_layout:
        fig.tight_layout()
    return fig


def showpsnr(rows, cols, imgs, reference_image=0, **kwargs):
    figs = dict()
    for i, im in enumerate(imgs):
        if i == reference_image:
            figs['clean'] = im
        else:
            figs[f"{i}: psnr={psnr(im, imgs[reference_image])}"] = im
    return show(rows, cols, figs, **kwargs)


def showpsnrd(rows, cols, imgdic, reference_image, psnr_in_title=True, **kwargs):
    figs = OrderedDict()
    figs['Clean'] = reference_image
    if not isinstance(imgdic, dict):
        imgdic = {i : im for i, im in enumerate(imgdic)}
    for key in imgdic:
        im = imgdic[key]
        axtitle = f"{key}, psnr={psnr(im, reference_image)}" if psnr_in_title else key
        figs[axtitle] = psimg(im, reference_image)
    return show(rows, cols, figs, **kwargs), figs
showps = showpsnrd

def showdenoise(rows: int, cols: int, clean : torch.tensor, noisy: torch.tensor, models: List[nn.Module],
                saves: os.PathLike = None, extra_name:str="",
                use_psimg:bool = True, **kwargs):
    """Shows all images in a figure.
    Takes the clean and noisy image and a list of models.
    Denoises the noisy image with the models and displays them.
    Additionally, the text on the images is created.
    The titles are determined by the models

    Args:
        rows (int): _description_
        cols (int): _description_
        clean (torch.tensor): _description_
        noisy (torch.tensor): _description_
        models (List[nn.Module]): _description_
        saves (os.PathLike, optional): _description_. Defaults to None.
        extra_name (str, optional): _description_. Defaults to "".
        use_psimg (bool, optional): _description_. Defaults to True.
    """
    def _psimg(im, cl):
        if use_psimg:
            return psimg(im, cl)
        else:
            return cv2.cvtColor(im[0].numpy(), cv2.COLOR_GRAY2RGB)

    figs = OrderedDict()
    figs['clean'] = clean
    figs['noisy'] = _psimg(noisy, clean)

    if saves is not None:
        if not os.path.isdir(saves):
            raise ValueError(f'{saves=} should be a directory but it is not.')
        cl = cv2.cvtColor(clean[0].numpy(), cv2.COLOR_GRAY2RGB)
        save_image(os.path.join(saves, f'{extra_name}clean.png'), cl)
        save_image(os.path.join(saves, f'{extra_name}noisy.png'), figs['noisy'])
    for mod in models:
        dim = mod.denoise(noisy)
        figs[mod.name] = _psimg(dim, clean)
        if saves is not None:
            save_image(os.path.join(saves, f'{extra_name}{mod.name}.png'), figs[mod.name])
    return show(rows, cols, figs, **kwargs)

def zoomshow(rows, cols, im, nim, cr, models, save_file =None, extension = "" ):
    if save_file is not None:
        save_image(os.path.join(save_file, f"{extension}clean.png"), cr(im))
        save_image(os.path.join(save_file, f"{extension}noisy.png"), cr(nim))
    ls = {'clean' : cr(im), 'noisy' : cr(nim)}
    for mod in models:
        z = cr(mod.denoise(nim))
        ls[mod.name] = z
        if save_file is not None:
            save_image(os.path.join(save_file, f"{extension}{mod.name}.png"), z)
    show(rows, cols, ls)

def add_gaussian_noise(img, stddev=1.):
    # noise = torch.zeros(img.shape).normal_(mean, stddev)
    noise = torch.normal(img.clone(), std=stddev)
    # out = img + noise
    return noise.clip_(0., 1.)

def salt_pepper(img, ps=0.02, pp=0.02):
    rand = torch.rand(img.shape)
    img = img.clone()
    img[rand < pp] = 0
    img[rand > (1-ps)] = 1.
    return img

def gauss_kernel(shape=(3, 3), sigma=0.5):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return torch.tensor(h.reshape(1, 1, *shape), dtype=torch.float)


gkernel = gauss_kernel(shape=(5, 5), sigma=1)
tconv = nn.functional.conv2d

def gauss_blurring(image):
    return tconv(image, gkernel, padding='same')

def psnr(im1: torch.tensor, im2: torch.tensor) -> torch.tensor:
    """Calculates the psnr between im1 and im2. It is expected that
    both images have values between 0 and 1
    and the shape has the structure:
    [color_channel, shape[0], shape[1]].

    It is possible to pass multiple channels in a tensor with shape
    [num_images, color_channels, shape[0], shape[1]].

    Then a list of psnr values is returned.

    Args:
        im1 (torch.tensor): First image (list)
        im2 (torch.tensor): second image (list)
    """
    # The following simultaneously checks if both arguments are tensors.
    if im1.shape != im2.shape:
        raise ValueError(
            f"im1 and im2 should have same shape. But im1.shape={im1.shape} != im2.shape={im2.shape}.")

    if not torch.is_tensor(im1):
        im1 = torch.tensor(im1)
    if not torch.is_tensor(im2):
        im2 = torch.tensor(im2)
    im1 = im1.detach()
    im2 = im2.detach()

    if len(im1.shape) == 4:
        mse = torch.sum(torch.square(
            im1 - im2), dim=[1, 2, 3]) / (torch.prod(torch.tensor(im1.shape[1:])))
    else:
        mse = torch.sum(torch.square(im1 - im2)) / \
            (torch.prod(torch.tensor(im1.shape)))

    mse[mse == 0] = torch.inf

    return 10*torch.log10(1/mse)


def psnrd(dc):
    return psnr(dc['clean'], dc['noisy'])



def convert_numpy(tensor : torch.tensor):
    if not torch.is_tensor(tensor):
        return np.array(tensor)
    x = tensor.to_numpy()
    if len(x.shape) == 3:
        return x[0]
    elif len(x.shape) == 4:
        return x[0,0]