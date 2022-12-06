# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 18:52:20 2022

@author: Anton

This is an implication of the Chambolle-Pock algorithm in pytorch.
This makes the algorithm available for cuda.
"""
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import misc
import dset
import models.reschapo as rcp
from misc import psimg, psnr, show, showpsnrd


def dis_divergence(du):
    """Calculates the discrete divergence of the input.

    Uses pytorch, if the input is on the gpu it is preserved resulting
    in a much faster execution."""

    d1, d2 = du.chunk(2, dim=1)
    # d1, d2 = du
    nz = torch.zeros_like(d1, device=du.device)

    nz[:, :, 1:-1, :] = d1[:, :, 1:-1, :]
    nz[:, :, 1:-1, :] -= d1[:, :, :-2, :]
    nz[:, :, 0, :] = d1[:, :, 0, :]
    nz[:, :, -1, :] -= d1[:, :, -2, :]

    nz[:, :, :, 1:-1] += d2[:, :, :, 1:-1]
    nz[:, :, :, 1:-1] -= d2[:, :, :, :-2]
    nz[:, :, :, 0] += d2[:, :, :, 0]
    nz[:, :, :, -1] -= d2[:, :, :, -2]
    return nz


def gradient(u):
    """Calculates the gradient of the image"""
    dd = torch.zeros((2, *u.shape), dtype=u.dtype)
    d1, d2 = dd
    # u_i,j+1 - u_i,j für j kleiner N.
    torch.subtract(u[1:, :], u[:-1, :], out=d1[:-1, :])
    # u_i+1,j - u_i,j für i kleiner M.
    torch.subtract(u[:, 1:], u[:, :-1], out=d2[:, :-1])
    return dd


def _gradient(u):
    """Calculates the gradient of the image"""
    b, c, h, w = u.shape
    dd = torch.zeros((b, 2*c, h, w), dtype=u.dtype, device=u.device)
    # u_i,j+1 - u_i,j für j kleiner N.
    dd[:, [0], :-1, :] = u[:, :, 1:, :] - u[:, :, :-1, :]
    # u_i+1,j - u_i,j für i kleiner M.
    dd[:, [1], :, :-1] = u[:, :, :, 1:] - u[:, :, :, :-1]
    return dd


def div(Du):
    # assert len(Du) == 2
    d1, d2 = Du
    # d1, d2 = Du.chunk(2,dim=1)
    dv = torch.zeros_like(d1)

    ## 1 - direction
    # 1 < i < N :  v_i,j,1 - v_i-1,j,1
    dv[1:-1, :] = d1[1:-1, :]
    dv[1:-1, :] -= d1[:-2, :]
    dv[0, :] = d1[0, :]  # i = 1
    dv[-1, :] -= d1[-2, :]  # i = N

    ## 2 - direction
    # 1 < j < M :
    dv[:, 1:-1] += d2[:, 1:-1]
    dv[:, 1:-1] -= d2[:, :-2]
    dv[:, 0] += d2[:, 0]  # j = 1
    dv[:, -1] -= d2[:, -2]  # j = M

    return dv


class PyChaPo(rcp.PDNet):
    """Implements the Chambolle-Pock algorithm. Can be inherited.
    Then only proximal operators need to be provided"""

    def __init__(self, sigma=None, tau=0.01, max_iter=150) -> None:
        super().__init__()
        if sigma is None:
            sigma = 1/(tau*8)
        self.sigma = sigma
        self.tau = tau
        self.max_iter = int(max_iter)
        self.name = "Generic Chambolle-Pock algorithm"

    def forward(self, im):
        if im.device != self.device:
            im = im.to(self.device)
        # xk = _xk = f = im
        xk = im
        _xk = im
        f = im
        b, c, h, w = im.shape
        yk = torch.zeros(b, 2*c, h, w, device=self.device)
        for i in range(self.max_iter):
            ykn = self.prox_F(yk + self.sigma*_gradient(_xk))
            xkn = self.prox_G(xk + self.tau*dis_divergence(ykn), f)
            _xkn = 2*xkn - xk
            yk = ykn
            xk = xkn
            _xk = _xkn
        return xk

    @torch.no_grad()
    def denoise_history(self, im, noisy):
        """Testing how the run behaves."""
        if self.device != im.device:
            im_device = im.device
            noisy = noisy.to(self.device)
        sh = noisy.shape
        if len(noisy.shape) == 2:
            # Expected: im.shape = (h,w)
            noisy = noisy[None, None, :, :]
        elif len(noisy.shape) == 3:
            # Expected: im.shape = (c,h,w)
            noisy = noisy[None, :, :]
        hist = torch.zeros(self.max_iter)
        xk = noisy
        _xk = noisy
        f = noisy
        b, c, h, w = noisy.shape
        yk = torch.zeros(b, 2*c, h, w, device=self.device)
        for i in range(self.max_iter):
            ykn = self.prox_F(yk + self.sigma*_gradient(_xk))
            xkn = self.prox_G(xk + self.tau*dis_divergence(ykn), f)
            _xkn = 2*xkn - xk
            yk = ykn
            xk = xkn
            _xk = _xkn
            hist[i] = psnr(_xk.clip(0, 1).reshape(
                sh).detach().to(im_device), im).mean()
        return _xk.clip(0, 1).reshape(sh).detach().to(im_device), hist

    @torch.no_grad()
    def denoise(self, im):
        if self.device != im.device:
            im_device = im.device
            im = im.to(self.device)
        sh = im.shape
        if len(im.shape) == 2:
            # Expected: im.shape = (h,w)
            im = im[None, None, :, :]
        elif len(im.shape) == 3:
            # Expected: im.shape = (c,h,w)
            im = im[None, :, :]
        it = self(im).clip(0, 1)
        # for i in range(self.max_iter):
        return it.reshape(sh).detach().to(im_device)

    @classmethod
    def denoise_(cls, im, **kwargs):
        """Denoise but without needing to instantiate a class."""
        model = cls(**kwargs)
        return model.denoise(im)

class Prox_l1(nn.Module):
    """Auxiliary class for LPychapo"""
    def __init__(self) -> None:
        super().__init__()
        self.t1 = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, xk):
        return torch.maximum(torch.minimum(self.t1, xk), -self.t1)

def _corr(inp, weights, padding):
    return F.conv2d(inp, weights, padding=padding)


def _conv(inp, weights, padding):
    return F.conv_transpose2d(inp, weights, padding=padding)

class LPychapo(rcp.PDNet):
    """A pytorch implementation of the TV regularization proposed by
    Kunisch and Pock in the paper
    "A Bilevel Optimization Approach for Parameter Learning in Variational Models", 2013

    The weights should be a list containing (k,k)-weights.
    The lamlist should be a list containing scalars.
    The list should be as long as the list of weights.

    Args:
        weights (torch.tensor): The weights for the convolution.
            You should give a padding when inserting not squared wweights.
        lamlist (List[torch.tensor], optional): A list of scalars. Defaults to None.
        padding (int, optional): The padding in the convolution.
            Defaults to the half size of the weight shape.
        sigma (float, optional): A scalar defining the sigma from the algorithm. Defaults to None.
        tau (float, optional): A scalar defining the tau from the algorithm. Defaults to 0.01.
        max_iter (int, optional): Number of iterations. Defaults to 150.
    """

    def __init__(self, weights: List[torch.tensor], lamlist: List[torch.tensor] = None,
                 padding: int = None, tau: float = 0.01, max_iter: int = 150) -> None:
        super().__init__()

        if lamlist is None:
            lamlist = np.zeros(len(weights)) + 1e-2

        if padding is None:
            padding = max(weights[0].shape)//2
        self.padding = padding

        assert len(weights) == len(lamlist)
        self.lenOP = len(weights)
        # self.lamlist = nn.ParameterList(nn.Parameter(torch.tensor(lam)) for lam in lamlist)
        self.lamlist = nn.Parameter(torch.tensor(
            lamlist, dtype=torch.float32).reshape(1, -1, 1, 1), requires_grad=True)
        self.weights = torch.cat([torch.tensor(w, dtype=torch.float32).reshape(
            1, 1, *w.shape) for w in weights], dim=1)

        self.tau = tau
        self.max_iter = int(max_iter)
        self.name = "L2-TV Chambolle-Pock with learnable lambdas and multiple weights"

        self.prox_F = Prox_l1()
        self.prox_G = rcp.Prox_l2(tau=tau)
        self.set_device(True)
        self.weights = self.weights.to(self.lamlist.device)

        self.arguments = dict(
            weights=weights,
            lamlist=lamlist,
            padding=padding,
            tau=tau,
            max_iter=max_iter,
        )

    def forward(self, im):
        if im.device != self.device:
            im = im.to(self.device)

        nweights = self.weights*self.lamlist
        sigma = 1/(self.tau*torch.norm(nweights, p=1)**2)

        xk = im
        _xk = im
        f = im
        b, c, h, w = im.shape
        yk = torch.zeros(b, self.lenOP*c, h, w, device=self.device)
        for _ in range(self.max_iter):
            ykn = self.prox_F(
                yk + sigma*_conv(_xk, nweights, padding=self.padding))
            xkn = self.prox_G(
                xk - self.tau*_corr(ykn, nweights, padding=self.padding), f)
            _xkn = 2*xkn - xk
            yk = ykn
            xk = xkn
            _xk = _xkn
        return xk


class L2ATV(rcp.PDNet):
    """Solves the model 1/2 * ||Ax - g||^2_2 + lam*TV(x)
    with the Chambolle-Pock algorithm.
    Primarily used for Deblurring if A is a blur operator.

    Here, A should be a convolution.

    Args:
        kernel (tensor): weights of the convolution A.
        lam (float): the
        padding (int): _description_
        tau (float, optional): _description_. Defaults to 0.01.
        sigma (_type_, optional): _description_. Defaults to None.
        max_iter (int, optional): _description_. Defaults to 150.
        use_cuda (bool, optional): _description_. Defaults to True.
    """

    def __init__(self, kernel, lam, padding, tau=0.01, sigma=None, max_iter=150,
                 use_cuda=True) -> None:

        super().__init__()
        if not torch.is_tensor(kernel):
            kernel = torch.tensor(kernel)
        if len(kernel.shape) != 4:
            if len(kernel.shape) == 2:
                kernel = kernel.reshape(1, 1, *kernel.shape)

        if sigma is None:
            sigma = 1/(tau*(torch.norm(kernel, p=1)**2 + 8))

        self.sigma = sigma
        self.tau = tau
        self.kernel = kernel

        self.proxF1 = rcp.Prox_l2T(sigma=sigma)
        self.proxF2 = rcp.Prox_l1(sigma=lam)
        self.max_iter = max_iter
        self.padding = padding

        self.set_device(use_cuda)
        self.kernel = self.kernel.to(self.device)

        self.arguments = dict(
            kernel=kernel,
            lam=lam,
            tau=tau,
            sigma=sigma,
            max_iter=max_iter,
            use_cuda=use_cuda,
        )
        self.name = "L2A-TV Chambolle-Pock solver"

    def forward(self, im):
        if im.device != self.device:
            im = im.to(self.device)
        b, c, h, w = im.shape

        yk_1 = torch.zeros_like(im)
        yk_2 = torch.zeros(b, 2*c, h, w, device=self.device)
        xk = im
        _xk = im
        f = im
        for k in range(self.max_iter):
            yk_1 = self.proxF1(yk_1 + self.sigma*_conv(_xk,
                               self.kernel, padding=self.padding), f)
            yk_2 = self.proxF2(yk_2 + self.sigma*_gradient(_xk))
            xkn = xk - self.tau * \
                (_corr(yk_1, self.kernel, padding=self.padding) - dis_divergence(yk_2))
            _xk = 2*xkn - xk
            xk = xkn
        return _xk


class PyChaPo2(rcp.PDNet):
    """Testing"""

    def __init__(self, sigma=None, tau=0.01, max_iter=150) -> None:
        super().__init__()
        if sigma is None:
            sigma = 1/(tau*8*0.01)
        self.sigma = sigma
        self.tau = tau
        self.lam = torch.nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.max_iter = int(max_iter)
        self.name = "L2-TV Chambolle-Pock with learnable lambda"

        self.prox_F = Prox_l1()
        self.prox_G = rcp.Prox_l2(tau=tau*self.lam)
        self.set_device(True)

    def forward(self, im):
        if im.device != self.device:
            im = im.to(self.device)
        # xk = _xk = f = im
        xk = im
        _xk = im
        f = im
        b, c, h, w = im.shape
        yk = torch.zeros(b, 2*c, h, w, device=self.device)
        for i in range(self.max_iter):
            ykn = self.prox_F(yk + self.sigma*self.lam*_gradient(_xk))
            xkn = self.prox_G(xk + self.tau*self.lam*dis_divergence(ykn), f)
            _xkn = 2*xkn - xk
            yk = ykn
            xk = xkn
            _xk = _xkn
        return xk

    @torch.no_grad()
    def denoise(self, im):
        if self.device != im.device:
            im_device = im.device
            im = im.to(self.device)
        sh = im.shape
        if len(im.shape) == 2:
            # Expected: im.shape = (h,w)
            im = im[None, None, :, :]
        elif len(im.shape) == 3:
            # Expected: im.shape = (c,h,w)
            im = im[None, :, :]
        it = self(im).clip(0, 1)
        # for i in range(self.max_iter):
        return it.reshape(sh).detach().to(im_device)

    @classmethod
    def denoise_(cls, im, **kwargs):
        model = cls(**kwargs)
        return model.denoise(im)


class L2TV(PyChaPo):
    """Implements lam/2||x-g||^2_2 + ||x'||_1"""

    def __init__(self, sigma=None, tau=0.01, lam=1/0.044, max_iter=150, use_cuda=True) -> None:
        super().__init__(sigma, tau, max_iter)
        self.prox_F = rcp.Prox_l1(sigma=1.)
        self.prox_G = rcp.Prox_l2_learnable(tau=tau*lam)

        self.set_device(use_cuda)
        self.name = "L2-TV Chambolle-Pock solver"


class L1TV(PyChaPo):
    """Implements lam||x-g||_1 + ||x'||_1"""

    def __init__(self, sigma=None, tau=0.02, lam=2.12, max_iter=150, use_cuda=True) -> None:
        super().__init__(sigma, tau, max_iter)
        self.prox_F = rcp.Prox_l1(sigma=1.)
        self.prox_G = rcp.Prox_l1_f(sigma=tau*lam)

        self.set_device(use_cuda)
        self.name = "L1-TV Chambolle-Pock solver"


class L1L2TV(PyChaPo):
    """Implements lam1||x-g||_1 + lam2/2||x-g||^2_2 + ||x'||_1"""

    def __init__(self, sigma=None, tau=0.02, lam1=1.5, lam2=2.9, max_iter=150, use_cuda=True) -> None:
        super().__init__(sigma, tau, max_iter)
        self.prox_F = rcp.Prox_l1(sigma=1.)
        self.prox_G = rcp.Prox_l1l2_f(tau=tau, lam1=lam1, lam2=lam2)

        self.set_device(use_cuda)
        self.name = "L1-L2-TV Chambolle-Pock solver"


def blur(img, kernel, padding):
    """Blurred ein Bild mit dem Kernel"""
    if not torch.is_tensor(kernel):
        kernel = torch.tensor(kernel)
    if len(kernel.shape) != 4:
        if len(kernel.shape) == 2:
            kernel = kernel.reshape(1, 1, *kernel.shape)
    return _conv(img, kernel, padding=padding)


def plot_analyze(nim, ref, xs, tvreg=L2TV, max_iter=150):
    from matplotlib import pyplot as plt

    psnrs = np.zeros_like(xs)
    ssims = np.zeros_like(xs)
    for i, x in enumerate(tqdm(xs)):
        dim = tvreg(lam=x, max_iter=max_iter).denoise(nim)
        psnrs[i] = psnr(dim, ref)
        ssims[i] = misc.torch_ssim(dim, ref)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 9))
    ax0.plot(xs, psnrs)
    ax1.plot(xs, ssims)

    ax0.set_title('PSNR')
    ax1.set_title('SSIM')
    # idy = losslist_detailed.argmin()
    # my = losslist_detailed[idy]
    # mx = lamlist_detailed[idy]

    # rx = max(lamlist_detailed.min(), lamlist.min())
    # rwidth = lamlist_detailed.max() - rx
    # ry = losslist_detailed.min()-10
    # rheight = losslist_detailed.max() - ry
    # rect = Rectangle((rx, ry), rwidth, rheight, linewidth=1,
    #                  edgecolor='g', facecolor='none', linestyle='dashed')
    # ax0.add_patch(rect)
    # ax0.plot([mx, mx], [0, my], color='green', linestyle='dashed')
    # ax0.plot([0, mx], [my, my], color='green', linestyle='dashed')
    # ax1.plot([mx, mx], [ry, my], color='green', linestyle='dashed')
    # ax1.plot([rx, mx], [my, my], color='green', linestyle='dashed')
    # ax1.set_xlim(rx, rx+rwidth)
    # ax1.set_ylim(ry, ry+rheight)

    # ax0.set_xlim(0, 1)
    # ax0.set_ylim(0, losslist.max())
    return psnrs, ssims


class InpaintingTv(PyChaPo):
    def __init__(self, sigma=None, tau=0.01, max_iter=150, use_cuda=True) -> None:
        super().__init__(sigma, tau, max_iter)
        self.prox_F = rcp.Prox_l1(sigma=1.)
        self.prox_G = rcp.Prox_indicator()

        self.set_device(use_cuda)
        self.name = "Inpainting-TV Chambolle-Pock solver"

    def forward(self, im):
        if im.device != self.device:
            im = im.to(self.device)
        # xk = _xk = f = im
        xk = im
        _xk = im
        f = im
        _s = im == 0
        b, c, h, w = im.shape
        yk = torch.zeros(b, 2*c, h, w, device=self.device)
        for _ in range(self.max_iter):
            ykn = self.prox_F(yk + self.sigma*_gradient(_xk))
            xkn = self.prox_G(xk + self.tau*dis_divergence(ykn), f, _s)
            _xkn = 2*xkn - xk
            yk = ykn
            xk = xkn
            _xk = _xkn
        return xk


def plot_images(clean, noise, denoise):
    """Auxiliary function to plot images."""
    showpsnrd(1, 3, {'noisy': noise, 'denoised': denoise}, clean, False)


def main():
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib import cm
    from tqdm import trange
    sns.set_theme()

    _l1 = False
    _l2 = False
    _l1l2 = False
    _trainExtendedTVReg = False

    max_iter = 150
    # Choose which noise type.
    # _, test_set = dset.getDefaultSet(transform=dset.NoCrop(stddev=25/255), max_test_size=50, max_train_size=1, preload=False)
    # _, test_set = dset.getDefaultSet(transform=dset.NoCropSP(), max_test_size=50, max_train_size=1, preload=False)
    # _, test_set = dset.getDefaultSet(transform=dset.NoCropSPGaussian(stddev=0.1), max_test_size=50, max_train_size=1, preload=False)
    # _, test_set = dset.getDefaultSet(transform=dset.CenterCrop(stddev=0.1, cropsize=128), max_test_size=5, max_train_size=1, preload=False)
    # im = test_set[25]
    # cim, nim = im['clean'], im['noisy']
    # print(f"Referenz: {psnr(cim,nim)}")
    # # psimg(nim, cim, save_path='noisy_test.png')

    train_set, test_set = dset.getDefaultSet(transform=dset.NoCrop(stddev=25/255),
                                             max_test_size=26, max_train_size=1, preload=False)
    im = test_set[25]
    cim, nim = im['clean'], im['noisy']
    cim, nim = dset.get_image(
        'schiff', dset.BlurNoise(stddev=0.05, sigma=(4, 4)))
    ref = cim
    # nim = cim.clone()
    # _s = torch.zeros_like(nim, dtype=torch.bool)
    # # _s.fill_(True)
    # _s[:, 100:104] = True
    # nim[_s] = 0

    # inp = InpaintingTv(max_iter=150)
    # dim = inp.denoise(nim)

    # show(1,3, [cim, nim, dim])
    scim, snim = dset.get_image(
        'schiff', transformation=dset.NoCrop(stddev=25/255))
    xs = np.linspace(0, 0.5, 100)
    # psnrs, ssims = plot_analyze(snim, scim, xs=xs, tvreg=L2ATV)

    kernel = np.ones((5, 5), dtype=np.float32)
    kernel /= kernel.sum()

    blurred = blur(scim, kernel, padding=2)

    psnrs = np.zeros_like(xs)
    ssims = np.zeros_like(xs)
    # ref = scim
    # nim = add_gaussian_noise(blurred, 0.025)
    # nim = blurred
    for i, x in enumerate(tqdm(xs)):
        dim = L2ATV(lam=x, kernel=kernel, padding=2,
                    max_iter=max_iter).denoise(nim)
        psnrs[i] = psnr(dim, ref)
        ssims[i] = misc.torch_ssim(dim, ref)

    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 9))
    # ax0.plot(xs, psnrs)
    # ax1.plot(xs, ssims)

    # ax0.set_title('PSNR')
    # ax1.set_title('SSIM')

    # psnrs = np.zeros_like(xs)
    # ssims = np.zeros_like(xs)
    # losses = np.zeros_like(xs)
    # ref = scim
    # nim = snim
    # for i, x in enumerate(tqdm(xs)):
    #     dim = L2TV(lam=x, max_iter=max_iter).denoise(nim)
    #     psnrs[i] = psnr(dim, ref)
    #     ssims[i] = misc.torch_ssim(dim, ref)
    #     losses[i] = (ref.ravel()**2 - dim.ravel()**2).sum()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 9))
    ax0.plot(xs, psnrs)
    ax1.plot(xs, ssims)

    ax0.set_title('PSNR')
    ax1.set_title('SSIM')

    print(
        f"Optimal pairs: \n\tlambda, psnr = f{(xs[psnrs.argmax()], psnrs.max())}\n\tlambda, ssim = f{(xs[ssims.argmax()], ssims.max())}")

    # opt_lam = xs[ssims.argmax()]
    # smoll_lam = 1/1e-3
    # big_lam = 1

    kernel = np.ones((5, 5), dtype=np.float32)
    kernel /= kernel.sum()

    _a1 = L2ATV(lam=xs[psnrs.argmax()], kernel=kernel,
                padding=2, max_iter=3000).denoise(nim)
    _a2 = L2ATV(lam=xs[ssims.argmax()], kernel=kernel,
                padding=2, max_iter=3000).denoise(nim)

    _a1_ = psimg(_a1, ref)
    _a2_ = psimg(_a2, ref)

    show(2, 2, {'clean': ref, 'noised': nim,
                'psnrmax': _a1_,
                'ssimmax': _a2_})

    # misc.save_image('verrauscht.png', snim)
    # misc.save_image('biglambda.png', L2TV(lam=big_lam, max_iter=1000).denoise(snim))
    # misc.save_image('kleinlambda.png', L2TV(lam=smoll_lam, max_iter=1000).denoise(snim))
    # misc.save_image('optimallambda.png', L2TV(lam=opt_lam, max_iter=1000).denoise(snim))

    if _trainExtendedTVReg:
        import get_convolution_kernels as gck

        tim = train_set[0]
        tcim, tnim = tim['clean'], tim['noisy']
        epochs = 750
        fit_args = dict(batch_size=1, epochs=epochs,
                        optimizer_args={'lr': 1e-3, 'betas': (0.9, 0.9)},
                        scheduler_args={
                            'T_max': epochs, 'eta_min': 1e-6},
                        tqdm_description="PyChaPo-Test",
                        test_set_validate_interval=50,
                        )

        # Choose a kernel set.
        # fps = gck.get_conv_matrices('fst_plus_snd')
        # fps = gck.get_conv_matrices('dct3')
        fps = gck.get_conv_matrices('dct5')
        kernels = [f['kernel'] for f in fps]
        model = LPychapo(kernels, max_iter=150, padding=2)
        hist, _ = model.conv_fit(train_size=1, test_size=15,
                                 test_transform=dset.NoCrop(stddev=25/255),
                                 train_transform=dset.NoCrop(stddev=25/255),
                                 fit_args=fit_args)
        print(f"model.lam={model.lamlist}")
        show(2, 3, [cim, psimg(nim, cim), psimg(model.denoise(nim), cim),
                    tcim, psimg(tnim, tcim), psimg(model.denoise(tnim), tcim)])
        model.save_model('dct5.pth')

    # L1-TV
    if _l1:
        lamlist1 = np.linspace(0, 4, num=250)
        psnrlist1 = np.zeros_like(lamlist1)
        for i, lam in enumerate(tqdm(lamlist1)):
            l1 = L1TV(tau=0.02, lam=lam, max_iter=max_iter)
            dim = l1.denoise(nim)
            psnrlist1[i] = psnr(cim, dim)
        fig, ax = plt.subplots(1, 1)
        ax.plot(lamlist1, psnrlist1)
        fig.savefig('l1tv.png')
        print(
            f"L1: Maximal möglich: {lamlist1[psnrlist1.argmax()]} mit einem PSNR={psnrlist1[psnrlist1.argmax()]}")
        l1 = L1TV(
            tau=0.02, lam=lamlist1[psnrlist1.argmax()], max_iter=max_iter)
        psimg(l1.denoise(nim), cim, save_path='l1_test.png')

    # L2-TV
    if _l2:
        lamlist2 = np.linspace(0, 1, num=1000)[1:]
        psnrlist2 = np.zeros_like(lamlist2)
        losslist = np.zeros_like(lamlist2)
        for i, lam in enumerate(tqdm(lamlist2)):
            l2 = L2TV(tau=0.01, lam=1/lam, max_iter=max_iter)
            dim = l2.denoise(nim)
            psnrlist2[i] = psnr(cim, dim)
            losslist[i] = (torch.norm(
                cim.ravel() - dim.ravel(), p=2)**2).item()
        fig, ax = plt.subplots(1, 1)
        ax.plot(lamlist2, losslist)
        fig.savefig('l2tv.png')
        print(
            f"L2: Maximal möglich: {lamlist2[psnrlist2.argmax()]} mit einem PSNR={psnrlist2[psnrlist2.argmax()]}")
        l2 = L2TV(
            tau=0.01, lam=lamlist2[psnrlist2.argmax()], max_iter=max_iter)
        psimg(l2.denoise(nim), cim, save_path='l2_test.png')

    # L1-L2-TV
    if _l1l2:
        nx, ny = 15, 500
        lgridx, lgridy = np.meshgrid(np.linspace(
            1, 2, num=nx), np.linspace(0, 10, num=ny), indexing='ij')
        psnrgrid12 = np.zeros_like(lgridx)
        for i in trange(nx):
            for j in range(ny):
                l1l2 = L1L2TV(
                    tau=0.02, lam1=lgridx[i, j], lam2=lgridy[i, j], max_iter=max_iter)
                dim = l1l2.denoise(nim)
                psnrgrid12[i, j] = psnr(cim, dim)
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        ax.plot_surface(lgridx, lgridy, psnrgrid12, cmap=cm.coolwarm)
        fig.savefig('l1l2tv.png')
        m = psnrgrid12.argmax()
        lx, ly = lgridx.ravel()[m], lgridy.ravel()[m]
        print(
            f"L1-L2: Maximal möglich: {lx=}, {ly=} mit PSNR={psnrgrid12.ravel()[m]}")
        l1l2 = L1L2TV(tau=0.02, lam1=lx, lam2=ly, max_iter=max_iter)
        psimg(l1l2.denoise(nim), cim, save_path='l1l2_test.png')


if __name__ == '__main__':
    main()
