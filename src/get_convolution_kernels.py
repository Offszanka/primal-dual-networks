# -*- coding: utf-8 -*-
from argparse import ArgumentError
import os
from math import cos, pi, sqrt

import numpy as np
from scipy import sparse
from scipy.signal import convolve

KERNEL_PATH = os.path.join('..', 'convolution_kernels')


def get_dct_kernel(u, v, n):

    def alpha(a):
        # Normalisation
        if a == 0:
            return sqrt(1.0/n)
        else:
            return sqrt(2.0/n)

    dc = np.zeros((n, n))

    for x in range(n):
        for y in range(n):
            dc[y, x] = alpha(u) * alpha(v) * cos(((2*x+1)*(u*pi)) /
                                                 (2*n)) * cos(((2*y+1)*(v*pi))/(2*n))
    return dc


def dct_kernels(n, fst=False, show=True):
    """
    returns all the dct kernels for n
    Only available for n in (3,5)!

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    fst : TYPE, optional
        Should I include the first (constant) basis vector?. The default is No.
    show : TYPE, optional
        show? The default is True.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    kernels : TYPE
        DESCRIPTION.

    """

    assert fst in (0, 1)

    if n == 3:
        rows = 1
        cols = 8
    elif n == 5:
        rows = 3
        cols = 8
    else:
        raise NotImplementedError('...')

    kernels = []
    for i in range(n):
        for j in range(n):
            if not fst:
                if i == 0 and j == 0:
                    continue
            kernels.append(get_dct_kernel(i, j, n))

    # if show:
    #     conv.plot_figures(rows, cols, kernels, show_axis=True)

    return kernels


def get_valid_coord(shape, co, kernel_size=3):
    """
    Gets a single coordinate co = (i,j).
    Returns all coordinates which are still in the image.
    I.e. (i-1,j-1), (i-1,j), ... , (i+1,j), (i+1,j+1)
    If i-1 or j-1 is negative or i+1 or j+1 is greater then f.shape then they are not returned.


    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    co : TYPE
        DESCRIPTION.

    Returns
    -------
    The surrounding coordinates, a mask indicating which the valid ones.

    """
    directions = np.linspace(-(kernel_size//2),
                             kernel_size//2, num=kernel_size, dtype=np.int)
    Z = np.asarray(np.meshgrid(directions, directions))
    Z = Z.reshape((2, -1)).T

    surr_coor = co + Z  # The surrounding coordinates

    mask_1 = np.all(surr_coor >= 0, axis=1)
    mask_2 = np.all(surr_coor < np.array(shape), axis=1)

    return surr_coor, mask_1 & mask_2


def get_matrix_representation(f, kernel):
    """
    Gets the (sparse) matrix representation of a convolution kernel.
    The kernel has to be a square matrix and the shape has to be odd.

    The goal is that G@f.ravel().reshape(f.shape)  == convolute(f, kernel, mode='same')

    TODO : Teste für andere kernel Größen.

    Parameters
    ----------
    f : np.ndarray
        The image which is only needed for the shape.
    kernel : np.ndarray
        The kernel matrix which is used in the convolution.

    Returns
    -------
    sparse matrix in lil_format.

    """
    assert kernel.shape[0] == kernel.shape[1] > 1  # kernel has to be at least 3 big and be a square matrix
    assert kernel.shape[0] % 2 == 1  # odd condition

    mat_repr = sparse.lil_matrix((f.size, f.size), dtype=f.dtype)
    kernel_size = kernel.shape[0]
    unravel_index = np.unravel_index
    ravel_multi_index = np.ravel_multi_index

    for row_ind in range(f.size):
        coor = unravel_index(row_ind, f.shape)
        surr_coor, valid = get_valid_coord(f.shape, coor, kernel_size)
        valid_coor = np.asarray([ravel_multi_index(ind, f.shape)
                                 for ind in surr_coor[valid]])
        # TODO: Warum ist das hier richtig? (zuerst valid_coor danach row_ind)
        mat_repr[valid_coor, row_ind] = kernel.T.ravel()[valid]

    return mat_repr


def teste_Gleichheit(mat: np.ndarray, f: np.ndarray, kernel: np.ndarray,
                     num: int = 1e2, verbose: bool = True):
    """
    Testet, ob mat wirklich die Matrix-Repräsentation von kernel ist,
    indem num viele zufällige Matrizen generiert werden und verglichen wird,
    ob sie das gleiche Resultat liefern.
    Getestet wird mit np.isclose

    Parameters
    ----------
    G : np.ndarray
        the matrix represantation of the kernel.
    f : np.ndarray
        The image which is only needed for the shape and size.
    kernel : np.ndarray
        the kernel.
    num : int, optional
        number of tests. The default is 1e2.
    verbose : bool, optional
        Show some text. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    def matrix_convolve(G, f):
        return (G@f.ravel()).reshape(f.shape)

    num = int(num)
    for i in range(num):
        z = np.random.random(f.shape)*100
        teste = matrix_convolve(mat, z)
        richtig = convolve(z, kernel,  mode='same')
        if not np.allclose(teste, richtig):
            if verbose:
                print(f"Fehler ab {i}:")
                print("z", z, "\nteste", teste, "\nrichtig", richtig)
            return False
    if verbose:
        print("Alles gleich!")
    return True


# 1st + 2nd - kernels.
kernel1 = np.asarray([[-1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.double)
kernel2 = np.asarray([[-1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.double)
kernel3 = np.asarray([[1, -2, 1], [0, 0, 0], [0, 0, 0]], dtype=np.double)
kernel4 = np.asarray([[1, 0, 0], [-2, 0, 0], [1, 0, 0]], dtype=np.double)
kernel5 = np.asarray([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.double)

kernel_ls= [kernel1, kernel2, kernel3, kernel4, kernel5]

available_kernels = {'fst_plus_snd': kernel_ls,
                     'dct3': dct_kernels(3, show=False),
                     'dct5': dct_kernels(5, show=False)}


def save_path(file: str, *,
              dir_path='convolution_kernels', img_size_path='256x256',
              filter_set_path='fst_plus_snd'):
    _, f_ext = os.path.splitext(file)

    if f_ext == '':
        file = '.'.join([file, 'npz'])
    elif f_ext != '.npz':
        raise ValueError(f'File extension {f_ext} is not supported!')

    path = os.path.join(dir_path, img_size_path, filter_set_path, file)
    if not os.path.exists(path):
        path = os.path.join('..', path)

    return path


def get_conv_matrices(kernel_names, img_size_path='256x256'):
    """
    Returns the kernel list given by the name of the set.
    A list with the dictionaries is returned.

    Parameters
    ----------
    kernel_names : TYPE
        DESCRIPTION.
    img_size_path : string, optional
        img sizes. Determines the shape of the resulting matrix.
        Of course, this matrix had to be calculated before at some point.
        The default is '256x256'.
    dtype : type, optional
        The dtype of the output.
        Currently this is only interesting if device='cuda'.
        Needed because float32 is much faster on gpu then float64 (double)

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    list of dictionaries.
        The matrix represantations of the kernels.
        The dictionaries contains the keys 'kernel' and 'matrix'.

    """
    if kernel_names not in available_kernels:
        raise ValueError(f'kernel set {kernel_names} not available. '
                         f'Try it with {list(available_kernels.keys())}.')

    def mdict(kern, mat_name):
        return {'kernel': kern,
                'matrix': sparse.load_npz(save_path(mat_name, img_size_path=img_size_path,
                                                    filter_set_path=kernel_names))}

    mats = []
    for k, kern in enumerate(available_kernels[kernel_names]):
        mat = mdict(kern, f'kernel{k}')
        mats.append(mat)

    return mats


def check_integrity(kernel_names, img_size_path='256x256'):
    """Checks if the saved matrix representation of the kernels of the kernel names
    correspond to the actual kernels.
    """

    img_size = int(img_size_path.split('x')[0])
    mats = get_conv_matrices(kernel_names, img_size_path)

    for k, mat in enumerate(mats):
        if teste_Gleichheit(mat['matrix'], np.zeros((img_size, img_size), dtype=np.double),
                            mat["kernel"], verbose=False):
            pass
        else:
            raise RuntimeError(f'{kernel_names} ergab einen Fehler bei '
                               f'{mat["kernel"]}, k = {k}')
    print(f"Stimmt so! (bei {kernel_names})")


def get_memory_space(array):
    """Returns and prints the memory space the data of Mk needs."""
    m_space = array.data.nbytes
    print(f"I need {np.round(m_space/1e3, 2)} KB space of data")
    return m_space


def construct_save_kernel_matrix(img, kernel, s_path, overwrite=False):
    """Generates and saves a matrix representation of a given kernel. The matrix representation
    adapts to the size of the given image.
    The matrix is saved in the directory KERNEL_PATH//s_path
    s_path should have .npz or nothing as extension.
    The directory in s_path should already be given.

    Args:
        img (np.ndarray): image to get the shape or other information.
        kernel (np.ndarray): The kernel of which the matrix representation is calculated.
        s_path (Path-like): Here the matrix rep is saved. If there is already

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """

    _, ext = os.path.splitext(s_path)
    if ext == '':
        s_path = s_path + '.npz'
    elif ext != '.npz':
        raise TypeError(f'File-ending {ext} is not supported.')
    s_path = os.path.join(KERNEL_PATH,  s_path)

    if os.path.isfile(s_path) and not overwrite:
        print("Found path-matching kernel. Loading this.")
        mat = sparse.load_npz(s_path)
        if not teste_Gleichheit(mat, img, kernel, 3, False):
            raise RuntimeError('Kernel and found matrix rep are not the same.')
    else:
        mat = get_matrix_representation(img, kernel).tocsr()
        if not teste_Gleichheit(mat, img, kernel, 10, False):
            raise RuntimeError('Kernel and matrix rep are not the same.')
        sparse.save_npz(s_path, mat)
    return mat
