# -*- coding: utf-8 -*-
"""
Another try to implement Kunisch&Pock learning.
This time with sparse matrices to make it faster.

Learning parameter for regularization to denoise images.
From Kunisch&Pock (2013) "A Bilevel Optimization Approach for Parameter Learning in Variational Models"

Here the Reduced Newton Learning is implemented (Algorithm 4.2 from the paper).

TODO das sauber Bild ist bisher self.img

Example usage:

    lam_list = np.load(os.path.join('data', 'found_lam.npy'))
    kernel_list = get_conv_matrices('fst_plus_snd')

    # To denoise:
    reg_l2 = KunischReg(img, kernel_list)
    sol = reg_l2(n_img, lam_list)

    # To learn:
    start_lam = np.random.random(len(kernel_list))
    learned_lam_list, learned_my_list = reg_l2.learn_params(lam0, n_img, max_iter=5, verbose=True)
"""

from time import perf_counter_ns as ptns
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import torch
import tqdm as tqdm
from scipy.sparse.linalg import LinearOperator, aslinearoperator, splu, spsolve

import dset
import misc
import newton_opt
# from ChaPo import L2K, Lorenz_L2TV_A, L2TV
from get_convolution_kernels import get_conv_matrices


def distance_kernel(v, A):
    """Returns inf_{x\in ker(A)} ||x - v||_2"""
    M = A.T@A
    sol = sparse.linalg.lsqr(M, v)[0]
    return np.linalg.norm(M@sol)


def build_matrix(op, shape=None):
    """Creates a matrix from a linear operator.
    Note that a dense matrix is created. (Not sparse!) """
    if shape is None:
        shape = op.shape
    I = np.eye(*shape)
    for j in range(I.shape[0]):
        I[:, j] = op(I[:, j])
    return I


def positive_approx(mat):
    """Makes an positive definite approximation of mat by flipping negative
    eigenvalues. The matrix should not be big (dense) and be symmetric."""
    if is_positive_definite(mat):
        return mat

    eig_vals, eig_vect = np.linalg.eigh(mat)
    return eig_vect @ np.diagflat(np.abs(eig_vals)) @ eig_vect.T


def is_positive_definite(mat):
    if mat.shape[0] != mat.shape[1]:
        return False
    try:
        np.linalg.cholesky(mat)
    except:
        return False
    return True


class Regularizer:
    def __init__(self, epsilon):
        self.eps = epsilon

    def reg_N(self, t):
        eps = self.eps
        t = np.abs(t)
        mask = t < eps  # Condition: |t| < eps
        t[mask] = - 1/(8*eps**3)*t[mask]**4 + 3/(4*eps)*t[mask]**2 + 3*eps/8

        return t

    def d_reg_N(self, t):
        eps = self.eps
        t_n = np.sign(t)
        mask = np.abs(t) < eps  # Condition: |t| < eps
        t_n[mask] = -1/(2*eps**3)*t[mask]**3 + 3/(2*eps)*t[mask]

        return t_n

    def dd_reg_N(self, t):
        eps = self.eps
        t_n = np.zeros_like(t)
        mask = np.abs(t) < eps  # Condition: |t| < eps
        t_n[mask] = -3/(2*eps**3) * t[mask]**2 + 3/(2*eps)  # else : t_n = 0

        return t_n

    def ddd_reg_N(self, t):
        eps = self.eps
        t_n = np.zeros_like(t)
        mask = np.abs(t) < eps  # Condition: |t| < eps
        t_n[mask] = -3/(eps**3) * t[mask]  # else : t_n = 0. Our Assumption.

        return t_n


class KunischPockLearner:
    def __init__(self, x, y, K_operators, regularizer_epsilon=1e-3,
                 name='Kunisch&Pock-Newton Learning'):
        """[summary]

        Args:
            x ([type]): noisy
            y ([type]): clean
            K_operators ([type]): [description]
            get_sol ([type], optional): [description]. Defaults to None.
            regularizer_epsilon ([type], optional): [description]. Defaults to 1e-3.
            name (str, optional): [description]. Defaults to 'Kunisch&Pock-Newton Learning'.
        """

        self.x = x
        self.y = y

        assert isinstance(K_operators, list)
        assert len(K_operators) > 0

        # K_operators = [{'matrix': sparse.block_diag(self.len_data*[op['matrix']],
        #                                             format='csc')}
        #                for op in K_operators]
        K_operators = [{'matrix': op['matrix']} for op in K_operators]
        self.K_operators = K_operators

        self.newton_solver = newton_opt.Newton(
            self._energy,
            self._grad_energy,
            self._hessian_energy,
            documentation=False,
            tol=1e-9*x.size,
            max_iter=100,
            sparse_hessian=True,
            solver=None
        )

        self.reg = Regularizer(regularizer_epsilon)

        self.history: Dict[str, np.ndarray] = dict()
        self.params = dict(
            name=name,
            dtype=np.double)

    def _energy(self, x):
        noisy = self.noisy
        clean = self.clean
        lam = self.lam
        assert x.size == noisy.size == clean.size

        x = x.ravel()

        reg_N = self.reg.reg_N

        summation = 0.5*np.linalg.norm(x.ravel() - noisy.ravel(), ord=2)**2

        for k, op_K in enumerate(self.K_operators):
            summation += lam[k] * (reg_N(op_K['matrix']@x)).sum()
        return summation

    def _grad_energy(self, x):
        noisy = self.noisy
        clean = self.clean
        lam = self.lam
        assert x.size == noisy.size == clean.size

        x = x.ravel()

        d_reg_N = self.reg.d_reg_N

        sum_vector = x - noisy.ravel()

        for k, op_K in enumerate(self.K_operators):
            mat = op_K['matrix']
            sum_vector += lam[k] * mat.T @ (d_reg_N(mat@x))

        return sum_vector

    def _hessian_energy(self, x):
        noisy = self.noisy
        clean = self.clean
        lam = self.lam
        assert x.size == noisy.size == clean.size

        x = x.ravel()

        dd_reg_N = self.reg.dd_reg_N

        sum_mat = sparse.eye(x.size, format='csc', dtype=self.params['dtype'])

        for k, op_K in enumerate(self.K_operators):
            mat = op_K['matrix']
            diag = sparse.diags(dd_reg_N(mat@x), format='csc')
            sum_mat += lam[k] * mat.T @ diag @ mat

        return sum_mat

    def get_reg_solution(self, noisy, lam_list, *,
                         clean=None,
                         verbose=False, start_x=None, **newton_args):
        assert len(lam_list) == len(self.K_operators), \
            f"Assert len(lam_list) == len(self.K_operators) but {len(lam_list)} != {len(self.K_operators)}"
        assert noisy.size == self.y.size

        self.lam = lam_list
        self.noisy = noisy
        self.clean = self.y
        if start_x is None:
            start_x = noisy
        sol = self.newton_solver(
            start_x.ravel(), verbose=verbose, **newton_args)
        self.newton_N = self.newton_solver.N

        sol = sol.reshape(noisy.shape)

        return sol

    def learn(self, lam0: List[float] = None, *,
              mu0: List[float] = None, max_iter: int = 1e2, tol: float = 1e-3, verbose: bool = True,
              ):
        """Learns the parameter for the saved operators and images.

        Args:
            lam0 (List[float], optional): a list or  an array of lambdas which are the initialized lambdas. Defaults to Array filled with 1e-2.
            mu0 (List[float], optional): A list with initialization values for mu. Defaults to Zero array.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1e2.
            tol (float, optional): The tolerance for G. Defaults to 1e-3.
            verbose (bool, optional): Enables/Disables the tqdm-progressbar. Defaults to True.

        Returns:
            np.ndarray: An array containing the optimized lambdas
        """

        max_iter = int(max_iter)
        if lam0 is None:
            lam0 = np.zeros(len(self.K_operators),
                            dtype=self.params['dtype']) + 1e-2
        if mu0 is None:
            mu0 = np.zeros_like(lam0)

        lam_k, mu_k = lam0.copy(), mu0.copy()
        q = len(lam_k)
        alpha = 0.9

        history = dict()
        history['lambda'] = np.zeros((max_iter, len(lam_k)), dtype=lam_k.dtype)
        history['mu'] = np.zeros((max_iter, len(mu_k)), dtype=lam_k.dtype)
        history['residual'] = np.zeros(max_iter, dtype=np.double)
        history['loss'] = np.zeros(max_iter)
        history['psnr'] = np.zeros(max_iter)
        history['reg_loss'] = np.zeros(max_iter)
        history['alpha'] = np.zeros(max_iter)

        self.history = history

        K_ops = self.K_operators
        d1_reg = self.reg.d_reg_N
        d2_reg = self.reg.dd_reg_N
        d3_reg = self.reg.ddd_reg_N

        mat_I = np.eye(q, dtype=np.double)
        noisy = self.x.ravel()
        xk = noisy.copy()
        clean_image = self.y.ravel()

        trange = tqdm.trange(max_iter, disable=not verbose,
                             desc='RNL-learning:')
        pdic = {'res': np.inf, 'loss': np.inf, 'psnr': 0, 'newton N': 0, }
        trange.set_postfix(pdic)
        k = max_iter

        timers = dict()
        timers['get_reg_solution'] = []
        timers['construction part matrices'] = []
        timers['inv l1'] = []
        timers['matrix p'] = []
        timers['rest'] = []
        for k in trange:
            timers['get_reg_solution'].append(ptns())
            xk = self.get_reg_solution(
                noisy.ravel(), lam_k, start_x=xk).ravel()
            timers['get_reg_solution'][-1] = ptns() - \
                timers['get_reg_solution'][-1]

            timers['construction part matrices'].append(ptns())
            # Constructing the part matrices
            mat_Q = sparse.eye(xk.size, dtype=np.double, format='csc')
            mat_R = np.zeros((xk.size, q), dtype=np.double)
            mat_L1 = sparse.eye(xk.size, dtype=np.double, format='csc')
            mat_L2 = np.zeros((xk.size, q), dtype=np.double)
            mat_M = np.diagflat(mu_k - lam_k > 0)

            for j in range(q):
                opK = K_ops[j]['matrix']

                R2 = sparse.diags(d2_reg(opK@xk), format='csc')
                mat_L1 += lam_k[j] * opK.T @ R2 @ opK
                mat_L2[:, j] = opK.T @ d1_reg(opK@xk)

                del opK, R2

            pk = spsolve(mat_L1, clean_image - xk)

            for j in range(q):
                opK = K_ops[j]['matrix']

                R2 = sparse.diags(d2_reg(opK@xk), format='csc')
                R3 = sparse.diags(d3_reg(opK@xk), format='csc')

                mat_R[:, j] = opK.T @ R2 @ opK @ pk
                diagKp = sparse.diags(opK@pk)
                mat_Q += lam_k[j] * opK.T @ R3 @ diagKp @ opK

            timers['construction part matrices'][-1] = ptns() - \
                timers['construction part matrices'][-1]
            timers['inv l1'].append(ptns())
            _lu = splu(mat_L1)
            invL1 = LinearOperator(mat_L1.shape, dtype=self.params['dtype'],
                                   matvec=_lu.solve)
            timers['inv l1'][-1] = ptns() - timers['inv l1'][-1]

            def linop_P(x):
                # fst = mat_L2.T @ mat_L1.inv @ mat_Q @ mat_L1.inv @ mat_L2
                fst = mat_L2 @ x
                fst = invL1 @ fst
                fst = mat_Q @ fst
                fst = invL1 @ fst
                fst = mat_L2.T @ fst
                # snd = - (mat_R.T @ invL1 @ mat_L2)
                snd = mat_L2 @ x
                snd = invL1 @ snd
                snd = mat_R.T @ snd
                snd = -snd
                # thd = - (mat_L2.T @ invL1 @ mat_R)
                thd = mat_R @ x
                thd = invL1 @ thd
                thd = mat_L2.T @ thd
                thd = -thd
                return fst + snd + thd

            timers['matrix p'].append(ptns())
            mat_P = build_matrix(linop_P, (q, q))
            mat_P = positive_approx(mat_P)
            timers['matrix p'][-1] = ptns() - timers['matrix p'][-1]
            timers['rest'].append(ptns())
            mat_S = np.block([[mat_P, -mat_I],
                              [mat_M, mat_I-mat_M]])

            # Build gk and solve for delta_k
            # g1 = mat_L1@pk + xk - clean_image
            g2 = pk.T @ mat_L2 - mu_k
            # g3 = xk - noisy + (lam_k*mat_L2).sum(axis = 1)
            g4 = mu_k - np.maximum(0., mu_k - lam_k)
            gk = np.hstack((g2, g4))

            delta_k = np.linalg.solve(mat_S, -gk)
            dl = delta_k[:q]
            dm = delta_k[q:]

            res = np.linalg.norm(gk)/gk.size
            loss = np.linalg.norm(xk - clean_image)**2
            psnr_val = psnr(xk, clean_image)
            pdic['res'] = res
            pdic['loss'] = loss
            pdic['newton N'] = self.newton_N

            if res < tol:
                trange.set_postfix(pdic)
                break

            index_I = mu_k - lam_k <= 0
            # derivative = 2*mat_L2.T@pk

            # def func(lam, **params):
            #     da = dl.copy()
            #     da[index_I] = lam
            #     xlam = self.get_reg_solution(
            #         noisy.ravel(), da, start_x=xk, **params).ravel()
            #     # Overall Energy
            #     return np.linalg.norm(xlam - clean_image)**2
            # if k <= 20:
            #     alpha = 0.0025
            # elif 20 < k:
            # alpha, armijo_n = newton_opt.armijo_line_search(func, derivative[index_I],
            #                                                 lam_k[index_I], dl[index_I],
            #                                                 fxk=np.linalg.norm(
            #                                                     xk - clean_image)**2,
            #                                                 max_iter=10,
            #                                                 func_params={'tol': 1e-9})
            # alpha = alpha * 0.9
            dl[index_I] = dl[index_I]

            # Step update
            lam_k += dl
            mu_k += dm

            # Update the history and the postfix dictionary.
            history['lambda'][k] = lam_k
            history['mu'][k] = mu_k
            history['loss'][k] = loss
            # history['psnr'][k] = psnr(self.y.ravel(), reconstructed)
            history['residual'][k] = res
            history['alpha'][k] = alpha

            pdic.update({'res': res, 'loss': loss, 'psnr': psnr_val})
            trange.set_postfix(pdic)
            timers['rest'][-1] = ptns() - timers['rest'][-1]

        if verbose:
            print(flush=True)
        self.iter_K = k

        print("Verbrauchte Zeiten")
        for key in timers:
            # print(f"{key} - vergangene Zeiten: {np.around(np.array(timers[key])/1e9, 3)}")
            print(f"\t {key} - Durschschnitt: {np.mean(timers[key])/1e9:.3f}")
            # print("Durchschnitte")

        for key in history:
            history[key] = history[key][:k]

        return lam_k

def psnr(im1, im2):
    import math
    im1 = im1.astype(np.double)
    im2 = im2.astype(np.double)
    this = 1*im1
    other = 1*im2
    assert this.shape == other.shape
    mse = np.sum(np.square(this - other)) / (np.prod(this.shape))
    if mse == 0:
        return np.inf

    return 10*math.log(1/mse, 10)


def save_image(name, im):
    return cv2.imwrite(f'{name}.png', (im*255).astype(np.uint8))


# def train_big_dataset(transform = None, file_path='___best_lambda.npy', iterations=10_000):
#     import os
#     import dset
#     import random

#     train_set, test_set = dset.getDefaultSet(transform=dset.StandardTransform(rcropsize=shape, stddev=25/255), max_test_size=200, max_train_size=200, preload=True)
#     cimg, nimg1 = test_set[25]['clean'].numpy()[0], test_set[25]['noisy'].numpy()[0]
#     cimg = cimg.astype(np.double)
#     print(f"{psnr(cimg, nimg1)=}")
#     alg_mat = get_nabla(cimg.shape)

#     lamlist = np.zeros(iterations)
#     # lamlist[:] = np.load('___best_lambda2.npy')
#     best_lambda = None
#     for i, im in enumerate(tqdm.trange(iterations, disable=False)):
#         set = random.choice((test_set, train_set))
#         zt = np.random.randint(0,len(set))
#         im = set[zt]

#         cim, nim = im['clean'], im['noisy']
#         cim = cim.numpy().astype(np.double)
#         nim = nim.numpy().astype(np.double)

#         kpl = KunischPockLearner(nim, cim,
#                                 [{'matrix': alg_mat}], 1e-3)
#         best_lambda = kpl.learn(max_iter=25, verbose=False, lam0=best_lambda)
#         lamlist[i] = best_lambda
#         if i % 100 == 0:
#             np.save(file_path, lamlist)

#     np.save(file_path, lamlist)
#     return lamlist

def train_single(cim, nim, kernelset = None, max_iter=100, tol=1e-3, ):
    """Find the optimzed lambda for a single image pair for a given set of kernels.

    Args:
        cim (np.ndarray): clean image
        nim (np.ndarray): nosiy image
        max_iter (int, optional): maximal number of iterations. Defaults to 100.
        tol (float, optional): Tolerance for norm(G). Defaults to 1e-3.
    """
    def n(im):
        return im.numpy()[0]
    cim = n(cim)
    nim = n(nim)
    if kernelset is None:
        kernelset = [{'matrix': get_nabla(cim.shape)}]

    kpl = KunischPockLearner(nim, cim,
                             kernelset, 1e-3)
    # kernels = get_conv_matrices('dct5', f"{nim.shape[0]}x{nim.shape[1]}")
    # kpl = KunischPockLearner(nim, cim, kernels, 1e-3)

    best_lambda = kpl.learn(max_iter=max_iter, verbose=True, lam0=None,
                            tol=tol)
    return best_lambda


def doc_single(cim: torch.tensor, nim: torch.tensor, kpl_learning_iterations: int = 5, big_len: int = 201, small_len: int = 101):
    """Documents the L2-TV behaviour of an image pair.
    Creats two plots and learns the best lambda.

    Args:
        cim (torch.tensor): The clean image.
        nim (torch.tensor): The noisy image.
        kpl_learning_iterations (int, optional): The maximal number of iterations for the Kunisch-Pock learning. Defaults to 5.
        big_len (int, optional): The grid size for the psnr-loss graph on the bigger intervall. Defaults to 201.
        small_len (int, optional): The grid size for the psnr-loss graph around the optimized parameter. Defaults to 101.

    Returns:
        Tuple[Figure, float]: The figure with the drawn stuff and the opzimied lambda.
    """
    import seaborn as sns
    from matplotlib.patches import Rectangle

    import PyChaPo as pcp
    sns.set_theme()
    nl1 = big_len
    nl2 = small_len
    kpl_iterations = kpl_learning_iterations

    def doc(lamls, lossls):
        for i, lam in enumerate(tqdm.tqdm(lamls)):
            l2 = pcp.L2TV(tau=0.01, lam=1/lam, max_iter=150)
            dim = l2.denoise(nim)
            lossls[i] = (torch.norm(cim.ravel() - dim.ravel(), p=2)**2).item()

    def draw(ax, ls, fls):
        for i in range(0, len(ls)-1):
            ax.plot([ls[i], ls[i+1]], [fls[i], fls[i+1]],
                    color=(1, 0, 0), marker='o', markersize=5,
                    markeredgecolor='k')

    lamlist = np.linspace(0, 1, num=nl1)[1:]
    losslist = np.zeros_like(lamlist)
    # np.save('lamlist', np.vstack((lamlist, losslist)))
    # np.save('lamlist_detailed', np.vstack((lamlist_detailed, losslist_detailed)))

    doc(lamlist, losslist)
    # KP-Learning

    def n(im):
        return im.numpy()[0]
    npcim = n(cim)
    npnim = n(nim)
    alg_mat = get_nabla(npcim.shape)
    kpl = KunischPockLearner(npnim[np.newaxis, :, :], npcim[np.newaxis, :, :],
                             [{'matrix': alg_mat}], 1e-3)
    best_lambda = kpl.learn(max_iter=kpl_iterations, verbose=True, lam0=None)

    # Documentation
    print(f"{best_lambda=} während {lamlist[losslist.argmin()]=}")
    ls = kpl.history['lambda']
    fls = np.zeros_like(ls)
    for i, lam in enumerate(ls):
        l2 = pcp.L2TV(tau=0.01, lam=1/lam, max_iter=150)
        dim = l2.denoise(nim)
        fls[i] = (torch.norm(cim.ravel() - dim.ravel(), p=2)**2).item()

    startpunkt = max(0, ls.min()*0.9)
    endpunkt = max(ls.max(), 0.06)
    lamlist_detailed = np.linspace(startpunkt, endpunkt, num=nl2)[1:]
    losslist_detailed = np.zeros_like(lamlist_detailed)
    doc(lamlist_detailed, losslist_detailed)
    print(
        f"{best_lambda=} während {lamlist_detailed[losslist_detailed.argmin()]=}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 9))
    ax0.plot(lamlist, losslist)
    ax1.plot(lamlist_detailed, losslist_detailed)
    idy = losslist_detailed.argmin()
    my = losslist_detailed[idy]
    mx = lamlist_detailed[idy]

    rx = max(lamlist_detailed.min(), lamlist.min())
    rwidth = lamlist_detailed.max() - rx
    ry = losslist_detailed.min()-10
    rheight = losslist_detailed.max() - ry
    rect = Rectangle((rx, ry), rwidth, rheight, linewidth=1,
                     edgecolor='g', facecolor='none', linestyle='dashed')
    ax0.add_patch(rect)
    ax0.plot([mx, mx], [0, my], color='green', linestyle='dashed')
    ax0.plot([0, mx], [my, my], color='green', linestyle='dashed')
    ax1.plot([mx, mx], [ry, my], color='green', linestyle='dashed')
    ax1.plot([rx, mx], [my, my], color='green', linestyle='dashed')
    ax1.set_xlim(rx, rx+rwidth)
    ax1.set_ylim(ry, ry+rheight)

    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, losslist.max())
    draw(ax1, ls, fls)
    draw(ax0, ls, fls)
    # fig.savefig('kplearning_l2tv_graph_iteration.png', dpi=100, bbox_inches='tight')
    return fig, best_lambda


def get_nabla(shape, format='csr'):
    """
    Calculates and returns the matrix representation
    of the discrete nabla operator as a sparse matrix.
    The shape is determined by the saved image.

    Returns
    -------
    nabla : sparse matrix in csr format
        Discrete derivative matrix representation of shape (2*img.size, img.size).
    """

    indx = shape[0]
    size = np.prod(shape)

    nabla_1 = sparse.diags(
        [[-1]*(size - indx) + [0]*indx, [1]*(size - indx) + [0]*indx], offsets=[0, indx])
    D = sparse.diags([[-1]*(indx-1) + [0], [1]*(indx-1) + [0]], offsets=[0, 1])
    nabla_2 = sparse.block_diag([D]*indx)
    nabla = sparse.vstack([nabla_1, nabla_2], format=format, dtype=np.double)

    return nabla


def train_single(cim, nim, weights, max_iter=5):
    cim = misc.convert_numpy(cim)
    nim = misc.convert_numpy(nim)
    kpl = KunischPockLearner(cim, nim, weights)
    lam = kpl.learn(max_iter=max_iter)

    return lam


if __name__ == '__main__':
    import os

    import torch
    shape = 64
    cim, nim = dset.get_image(
        'schiff', transformation=dset.CenterCrop(cropsize=shape, stddev=25/255))
    # fig, blam = doc_single(cim, nim)

    kpl = KunischPockLearner(nim.numpy()[:, :], cim.numpy()[:, :],
                             get_conv_matrices('dct5', '64x64'), 1e-3)
    blambda = kpl.learn()
