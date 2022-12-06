# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:43:38 2022

@author: Anton

Several classed and methods for the dataset as well as some transformation functions
for convenient loading and noise adding.
"""

import os
import warnings
from os import PathLike
from typing import Callable, Tuple

import torch
import torchvision
from torchvision import io
from torch.utils.data import Dataset

from misc import add_gaussian_noise, salt_pepper

path_nets = os.path.join('curious_networks')
path_succ = os.path.join(path_nets, 'successfull_Networks')


class IBlubbSet(Dataset):
    """Toy Dataset.
    Simply provide a clean-noisy image pair. They are returned then later"""

    def __init__(self, image_set) -> None:
        """

        Args:
            images (torch.tensor): _description_
            transform (Callable, optional): _description_. Defaults to None.
            stddev (float, optional): _description_. Defaults to 25/255.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        self.image_set = image_set

    def __len__(self):
        return len(self.image_set['clean'])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        clean = self.image_set['clean'][index]
        noisy = self.image_set['noisy'][index]
        return {'clean': clean, 'noisy': noisy}


class ImageSet(Dataset):
    """Creates an Image Dataset.

    Args:
        dir_path (Path): The path to the directory which contains the images.
        transform (Callable, optional): Custom transformations after the image is loaded.
            Defaults to None which is the identity.
            It is expected that the image takes the freshly loaded image and outputs two images in a tuple.
        max_size (int, optional): The maximum size of the dataset (includes both the train and test set).
            Defaults to None.

    Raises:
        ValueError: If dir_path does not point to a directory.
    """
    def __init__(self, dir_path: PathLike, transform: Callable = None,
                 max_size: int = None) -> None:

        super().__init__()
        self.dir_path = dir_path
        if transform is None:
            def transform(x):
                return x, x

        self.transform = transform
        if not os.path.isdir(dir_path):
            raise ValueError(
                f"dir_path should direct to a directory but it is not!\n\t{dir_path=}")
        dps = os.listdir(dir_path)
        if len(dps) == 0:
            raise ValueError(f"The directed path\n\t{dir_path}\n\t does not contain any files.")
        if max_size is not None:
            if len(dps) < max_size:
                warnings.warn(
                    f'{max_size=}>size of set={len(dps)}. This can lead to incorrect behaviour if you trust on your choice of max_size.')
            dps = dps[:max_size]
        self.dps = dps

    def __len__(self):
        return len(self.dps)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        ipath = os.path.join(self.dir_path, self.dps[index])
        img = io.read_image(ipath)
        img, nimg = self.transform(img)
        return {'clean': img, 'noisy': nimg}


class PreloadImageSet(ImageSet):
    """Creates a DataSet. Preloads all the images which makes it faster
    when going frequently over the dataset."""

    def __init__(self, dir_path: PathLike, transform: Callable = None, max_size: int = None,
                 loading_transform: Callable = None) -> None:
        super().__init__(dir_path, transform, max_size)
        length = len(self.dps)
        ipath = os.path.join(self.dir_path, self.dps[0])
        img = io.read_image(ipath)
        self.ddset = torch.zeros((length, *img.shape))
        for i in range(length):
            ipath = os.path.join(self.dir_path, self.dps[i])
            im = io.read_image(ipath)
            if loading_transform is None:
                def loading_transform(x):
                    if x.shape != img.shape:
                        return x.transpose(1, 2)
                    else:
                        return x
            self.ddset[i] = loading_transform(
                im)  # Let Python handle the errors

    def __getitem__(self, index):
        img = self.ddset[index]
        img, nimg = self.transform(img)
        return {'clean': img, 'noisy': nimg}


def getTrainTestSet(path_train: PathLike, path_test: PathLike, transform: Callable = None,
                    train_transform: Callable = None, test_transform: Callable = None,
                    max_train_size: int = None, max_test_size: int = None, preload=False) -> Tuple[ImageSet, ImageSet]:
    """Auxiliary function. Takes paths to the training and test dataset and returns both."""
    isetloader = PreloadImageSet if preload else ImageSet
    if test_transform is None:
        test_transform = transform
    if train_transform is None:
        train_transform = transform
    train_set = isetloader(
        path_train, max_size=max_train_size, transform=train_transform)
    test_set = isetloader(
        path_test, max_size=max_test_size, transform=test_transform)

    return train_set, test_set


# def getDefaultSet(transform: Callable = None,
#                   train_transform: Callable = None, test_transform: Callable = None,
#                   max_train_size: int = None, max_test_size: int = None, preload=False):
#     """Returns the default data set. """
#     path_train = os.path.join('..', 'images', 'dataset', 'BSR_images', 'train')
#     path_test = os.path.join('..', 'images', 'dataset', 'BSR_images', 'test')
#     if not os.path.exists(path_train) or len(os.listdir(path_train)) == 0:
#         raise ValueError(
#             f"There is no dataset at {path_train}\nConsider to place some images there.")
#     if not os.path.exists(path_test) or len(os.listdir(path_test)) == 0:
#         raise ValueError(
#             f"There is no dataset at {path_test}\nConsider to place some images there.")
#     return getTrainTestSet(
#         path_train=path_train,
#         path_test=path_test,
#         max_train_size=max_train_size,
#         max_test_size=max_test_size,
#         transform=transform,
#         train_transform=train_transform,
#         test_transform=test_transform,
#         preload=preload,
#     )


def load_image(path, transform):
    im = io.read_image(path)
    return transform(im)


class StandardTransform():
    """Crops randomly a patch from the image and adds Gaussian noise to it."""
    def __init__(self, rcropsize=256, stddev=25/255) -> None:
        self.Gray = torchvision.transforms.Grayscale()
        self.RCrop = torchvision.transforms.RandomCrop(rcropsize)
        self.stddev = stddev

    def __call__(self, im) -> torch.Tensor:
        im = self.Gray(im)
        im = self.RCrop(im).type(torch.float)/255
        nim = add_gaussian_noise(im, stddev=self.stddev)
        return im, nim


RandomCrop = StandardTransform


class NoCrop():
    """Adds Gaussian noise to an image."""
    def __init__(self, stddev=25/255) -> None:
        self.Gray = torchvision.transforms.Grayscale()
        self.stddev = stddev

    def __call__(self, im) -> torch.Tensor:
        im = self.Gray(im).type(torch.float)/255
        nim = add_gaussian_noise(im, stddev=self.stddev)
        return im, nim


class NoCropSP():
    """Adds salt-pepper noise to an image."""
    def __init__(self, pp=0.02, ps=0.02) -> None:
        self.Gray = torchvision.transforms.Grayscale()
        self.pp = pp
        self.ps = ps

    def __call__(self, im) -> torch.Tensor:
        im = self.Gray(im).type(torch.float)/255
        nim = salt_pepper(im, pp=self.pp, ps=self.ps)
        return im, nim


class NoCropSPGaussian():
    """Crops randomly a patch from the image and adds salt-pepper noise to it."""
    def __init__(self, stddev=0.05, pp=0.02, ps=0.02) -> None:
        self.Gray = torchvision.transforms.Grayscale()
        self.pp = pp
        self.ps = ps
        self.stddev = stddev

    def __call__(self, im) -> torch.Tensor:
        im = self.Gray(im).type(torch.float)/255
        nim = add_gaussian_noise(im, stddev=self.stddev)
        nim = salt_pepper(nim, pp=self.pp, ps=self.ps)
        return im, nim


class CenterCrop():
    """Crops a part of the center and adds Gaussian noise to it."""
    def __init__(self, cropsize=256, stddev=25/255) -> None:
        self.Gray = torchvision.transforms.Grayscale()
        self.CCrop = torchvision.transforms.CenterCrop(cropsize)
        self.stddev = stddev

    def __call__(self, im) -> torch.Tensor:
        im = self.Gray(im)
        im = self.CCrop(im).type(torch.float)/255
        nim = add_gaussian_noise(im, stddev=self.stddev)
        return im, nim


class BlurCrop():
    """Crops a patch of the image and blurres it with Gaussian Blur"""
    def __init__(self, rcropsize=256, kernel_size=5, sigma=(0.1, 4)) -> None:
        self.Gray = torchvision.transforms.Grayscale()
        self.CCrop = torchvision.transforms.RandomCrop(rcropsize)
        self.blur = torchvision.transforms.GaussianBlur(
            kernel_size, sigma=sigma)

    def __call__(self, im) -> torch.Tensor:
        im = self.Gray(im)
        im = self.CCrop(im).type(torch.float)/255
        nim = self.blur(im)
        return im, nim


class BlurCropNoise():
    """Crops a patch of the image, blurres it with Gaussian Blur and adds Gaussian noise."""
    def __init__(self, rcropsize=256, kernel_size=5, sigma=(0.1, 4), stddev=0.05) -> None:
        self.Gray = torchvision.transforms.Grayscale()
        # self.RCrop = torchvision.transforms.RandomCrop(rcropsize)
        self.blur = torchvision.transforms.GaussianBlur(
            kernel_size, sigma=sigma)
        self.stddev = stddev
        self.rcropsize = (rcropsize, rcropsize)
        self.crop = torchvision.transforms.functional.crop

    def __call__(self, im) -> torch.Tensor:
        im = self.Gray(im).type(torch.float)/255
        nim = self.blur(im)

        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            im, self.rcropsize)

        im = self.crop(im, i, j, h, w)
        nim = self.crop(nim, i, j, h, w)
        nim = add_gaussian_noise(nim, stddev=self.stddev)
        return im, nim


class BlurNoise():
    """Blurrs the image and adds noise to it."""
    def __init__(self, kernel_size=5, sigma=(0.1, 4), stddev=0.05) -> None:
        self.Gray = torchvision.transforms.Grayscale()
        self.blur = torchvision.transforms.GaussianBlur(
            kernel_size, sigma=sigma)
        self.stddev = stddev

    def __call__(self, im) -> torch.Tensor:
        im = self.Gray(im).type(torch.float)/255
        nim = self.blur(im)
        nim = add_gaussian_noise(nim, stddev=self.stddev)
        return im, nim


class Blur():
    """Blurrs the image."""
    def __init__(self, rcropsize=256, kernel_size=5, sigma=(0.1, 4)) -> None:
        self.Gray = torchvision.transforms.Grayscale()
        self.CCrop = torchvision.transforms.RandomCrop(rcropsize)
        self.blur = torchvision.transforms.GaussianBlur(
            kernel_size, sigma=sigma)

    def __call__(self, im) -> torch.Tensor:
        im = self.Gray(im).type(torch.float)/255
        nim = self.blur(im)
        return im, nim


def get_image(img_name=None, transformation=None, to_numpy=False):
    """Loads an image from the example_images folder,
    transforms it with the given transformation and returns it.

    Args:
        img_name (str, optional): The name of the image. The extension '.jpg' is conveniently
            added if not given. Defaults to None.
        transformation (Callable, optional): A transformation to apply to the image.
            Should return a tuple containing the clean and transformed image. Defaults to NoCrop.
        to_numpy (bool, optional): Determines if the resulting array should be
            transformed to a numpy array. Defaults to False.

    Returns:
        The result of the transformation.
    """
    if img_name is None:
        img_name = "schiff"
    if transformation is None:
        transformation = NoCrop()

    _, ext = os.path.splitext(img_name)
    if ext == '':
        img_name = img_name + '.jpg'

    im = io.read_image(os.path.join('..', 'toy_images', img_name))
    im = transformation(im)
    if to_numpy:
        im, nim = im
        im = im.numpy()[0]
        nim = nim.numpy()[0]
        im = im, nim
    return im


def r(im):
    """Auxiliary test function. Expects a dict with 'noisy' and 'clean' as keys."""
    return im['clean'], im['noisy']
