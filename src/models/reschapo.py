# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:27:37 2022

@author: Anton

Implements the foundation class for all models.
It implements the fit method which every model uses.
"""
import math
import os
import re
import warnings
from collections import namedtuple
from datetime import datetime
from os import PathLike
from typing import Dict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, RandomSampler

import dset
from misc import psnr


PATH_TRAIN = os.path.join('..', 'images', 'dataset', 'BSR_images', 'train')
PATH_TEST = os.path.join('..', 'images', 'dataset', 'BSR_images', 'test')

TabloidInfo = namedtuple("TabloidInfo", "layer kernelsize channelsize shared")


def load_network(path, map_location=None):
    """Tries to load the network from the given path.
    tries to load the data from the path and from the directory 'curious_networks' if not found there.

    Args:
        path (PathLike): A path leading to the file containing the network.
        map_location (str, optional): map_location passed to torch.load.

    Returns:
        The result of the torch.load method onto the given file.
    """
    if os.path.isdir(path):
        # Get the latest modified checkpoint of the saved checkpoint-epochs.
        path = max([os.path.join(path, p) for p in os.listdir(
            path) if p.endswith('.pth')], key=os.path.getmtime)
    try:
        cp = torch.load(path, map_location=map_location)
    except FileNotFoundError:
        # Try to load from curious_network dir if file not found. Maybe there is a network.
        # Otherwise let pytorch handle the error
        cp = torch.load(os.path.join('curious_networks', path),
                        map_location=map_location)
    return cp


def plot_history(hist, ref_loss=None, ref_psnr=None, title=None,
                 train_color='b', test_color='g', ref_color='r',
                 vert_line=None, save_file_path=None):
    """Plots the history given in hist.
    hist must contain the keys loss and psnr. Optionally, it can contain
    the keys test_loss and test_psnr. Then these are also plotted.

    Args:
        hist (dict): Dictionary containing the information about the history of the
            learning adventure.
        ref_loss (float, optional): A reference loss, plots a horizontal line in the
            loss panel. Defaults to None.
        ref_psnr (float, optional): A reference psnr, plots a horizontal line in the
            psnr panel. Defaults to None.
        title (str, optional): The title of the figure. Defaults to None.
        train_color (str, optional): The color of the training plots. Defaults to 'b'.
        test_color (str, optional): The color of the test plots. Defaults to 'g'.
        ref_color (str, optional): The color of the reference values. Defaults to 'r'.
        vert_line (_type_, optional): A dictionary containing information about the added vertical
            line. It is passed as ax.axvline(**ver_line). Defaults to no vertical line.
        save_file_path (Pathlike, optional): A path where a file should be save.
            If not given the figure is not saved.
    """
    fig, axs = plt.subplots(1, 2, squeeze=False,)
    axs.ravel()[0].plot(hist['loss'], c=train_color, label="loss")
    if 'test_loss' in hist:
        axs.ravel()[0].plot(hist['test_loss'], c=test_color, label="test-loss")
        axs.ravel()[0].legend()
    if ref_loss is not None:
        axs.ravel()[0].axhline(ref_loss, c=ref_color)
    axs.ravel()[0].set_title('loss')

    axs.ravel()[1].plot(hist['psnr'], c=train_color, label="psnr")
    if 'test_psnr' in hist:
        axs.ravel()[1].plot(hist['test_psnr'], c=test_color, label="test-psnr")
        axs.ravel()[1].legend()
    if ref_psnr is not None:
        axs.ravel()[1].axhline(ref_psnr, c=ref_color)
    axs.ravel()[1].set_title('psnr')

    if vert_line is not None:
        axs.ravel()[0].axvline(**vert_line)
        axs.ravel()[1].axvline(**vert_line)

    fig.suptitle(title)
    if save_file_path is not None:
        fig.savefig(save_file_path)


def get_last_checkpoint(dir_path, get_file_list=False):
    """Looks into the given path of the directory and returns the path leading to the latest
    checkpoint.

    The 'latest' is determined by the name of the file.
    If get_file_list is True the list of files in the directory is also returned.

    Args:
        dir_path (Pathlike): Path leading to the directory where the checkpoints are saved.
        get_file_list (bool, optional): Determines if the file list should also be returned.
            Defaults to False.

    Raises:
        ValueError: If the path does not lead to a directory.

    Returns:
        The path leading to the latest checkpoint. Determined by the number inside the file name.
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"Please give a path to a directory, {dir_path}")
    paths = [os.path.join(dir_path, p)
             for p in os.listdir(dir_path) if p.endswith('.pth')]
    # It is expected that a checkpoint has the form ...\\epoch_xxx.pth
    # where xxx is an abitrary non negative integer
    epoch_lists = [(p, re.findall('epoch_\d+', p)) for p in paths]
    epoch_lists = [(ep[0], int(ep[1][0].split('_')[-1]))
                   for ep in epoch_lists if len(ep[1]) > 0]

    if len(epoch_lists) > 0:
        last_epoch = max(epoch_lists, key=lambda x: x[1])
    else:
        last_epoch = [None]

    if get_file_list:
        return last_epoch[0], epoch_lists
    return last_epoch[0]


class PDNet(nn.Module):
    """Implements the base class for the used networks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.arguments = None
        self.device = "cpu"

    def set_device(self, use_cuda: bool = True):
        """Sends this file to cuda if wished."""
        if use_cuda:
            if torch.cuda.is_available():
                if torch.cuda.device_count() >= 2:
                    # If this script is runned on the cloud machine take the second graphic card.
                    self.to("cuda:1")
                    self.device = "cuda:1"
                else:
                    self = self.to("cuda")
                    self.device = "cuda"
            else:
                raise ValueError(
                    "No cuda found. Use use_cuda=False instead to use the cpu.")
        else:
            self.device = "cpu"
        return self.device

    @staticmethod
    def plot_history_from_path(path, **kwargs):
        """Plots the history given in the path. See plot_history for more information."""
        cp = load_network(path, 'cpu')
        plot_history(cp, **kwargs)

    def _load(self, path: PathLike, *, data: Dict = None) -> None:
        """Loads the model of the given path which leads to a pytorch model.
        """
        if data is None:
            # To prevent additional loadings of the same file
            data = load_network(path, map_location=self.device)
        if 'model_state_dict' in data:
            # This indicates that more parameters were saved than just the model_state_dict.
            try:
                print(self.load_state_dict(data['model_state_dict']))
            except RuntimeError as rerr:
                descr = data.get('description')
                if descr is not None:
                    raise RuntimeError(
                        f"Maybe the description can help:\n{descr}") from rerr
                raise rerr
            # self.update_constants(**data)
        else:
            # Try this out if data only contains the parameter dict.
            # Let pytorch handle the error if this is not the case.
            print(self.load_state_dict(data))

    @classmethod
    def parse_model(cls, path, use_cuda=None):
        """Parses the model from the given path.
        Finds the correct parameters from additional info saved in the file.
        So, if the file was not saved with this information this method raises an errror.

        Args:
            path (Pathlike): Path leading to the file containing the network.
            use_cuda (bool, optional): Determines if this model should be loaded
            onto the graphics card. Defaults to None.

        Raises:
            ValueError: If the file does not contain the parameters.

        Returns:
            The loaded model.
        """
        tdic = torch.load(path, 'cpu')
        if tdic.get('arguments') is None:
            raise ValueError(
                f"{path} did not save the arguments. I am helpless. :(")

        if use_cuda is not None:
            tdic['use_cuda'] = use_cuda

        instance = cls(**tdic['arguments'])
        instance._load(path)
        return instance

    def save_model(self, path: PathLike, **values):
        """Saves the model to the given path.
        You can add further data to save by adding keyword arguments.

        Args:
            path (PathLike): Path were the model should be saved.
        """
        if self.arguments is None:
            print("Please save the class initialization arguments for this model. "
                  "This helps to parse the model later.")
        torch.save(dict(model_state_dict=self.state_dict(),
                        arguments=self.arguments,
                        **values),
                   path)

    def save_checkpoint(self, path: PathLike, history: Dict):
        """Saves the checkpoint in the path."""
        self.save_model(path, epoch=len(history['loss']),
                        **history)

    def checkpoint_organizer(self, path, run_idx, current_epoch, optimizer, scheduler,
                             history, description=None, file_name: str = None, max_number_saves=3):
        """Wrapper to organize the checkpoint saves in a single folder.

        Args:
            path (_type_): a path leading to the dictionary where the folder should be created.
            epoch (_type_): _description_
        """
        dir_path = os.path.join(path, str(run_idx))
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        if description is not None:
            descr_path = os.path.join(dir_path, 'description.txt')
            if not os.path.isfile(descr_path):
                with open(descr_path, 'w', encoding='utf-8') as f:
                    f.write(f"{self.name}\n\n" + "="*50 + '\n\n')
                    f.write(description)

        _, flist = get_last_checkpoint(dir_path, True)
        flist = sorted(flist, key=lambda x: x[1])
        for fpath, _ in flist[:-max_number_saves]:
            os.remove(fpath)

        file_name = f"epoch_{current_epoch}.pth" if file_name is None else file_name
        file_path = os.path.join(dir_path, file_name)
        self.save_model(file_path, optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(), current_epoch=current_epoch,
                        arguments=self.arguments, **history)

    def load_checkpoint(self, path, optimizer=None, scheduler=None, history=None):
        """Modifies the given arguments inplace to match the latest checkpoint given in path."""
        if os.path.isdir(path):
            _path = get_last_checkpoint(path)
            if _path is None:
                raise FileNotFoundError(
                    f"There does not exist a checkpoint at {_path}.")
            path = _path
        checkpoint = load_network(path, map_location=self.device)
        curr_epoch = checkpoint['current_epoch']

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())
            warnings.warn(
                "No optimizer is given. Will load ADAM instead. May lead to errors.")

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.load_state_dict(checkpoint['model_state_dict'])

        def __cget(key):
            return checkpoint[key][:curr_epoch]
        if history is not None and len(history['loss']) > curr_epoch:
            history['loss'][:curr_epoch] = __cget('loss')
            history['psnr'][:curr_epoch] = __cget('psnr')
            history['test_loss'][:curr_epoch] = __cget('test_loss')
            history['test_psnr'][:curr_epoch] = __cget('test_psnr')
        return {'current_epoch': curr_epoch, 'optimizer': optimizer, 'scheduler': scheduler}

    @torch.no_grad()
    def denoise(self, x):
        """Denoisese the given input by changing its shape to an approriate format."""
        shape = x.shape
        if len(x.shape) == 2:
            x = x[None, None]
        if len(x.shape) == 3:
            x = x[None]
        out = self(x.to(self.device))
        out.clip_(0., 1.)
        return out.reshape(shape).to(x.device)

    def forward(self, x):
        raise NotImplementedError(
            "Inherit this class and implement this method.")

    @torch.no_grad()
    def validate(self, dataset, fn_loss=None, batch_size=25):
        """Calculates the loss and psnr values for the giventest data set. (If available)

        Args:
            dataset (dset.ImageSet): The test data set.
            fn_loss: The loss function. Default is the MSELoss.
            batch_size (int, optional): The batch size to check the test data. Defaults to 32.
        """
        if fn_loss is None:
            fn_loss = nn.MSELoss()

        training = self.training
        if training:
            self.eval()
        dl = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

        tloss = 0.
        tpsnr = 0.
        for batch in dl:
            x = batch['noisy'].to(self.device)
            y = batch['clean'].to(self.device)
            pred = self(x)
            loss = fn_loss(pred, y)
            tloss += loss.item()
            tpsnr += psnr(pred, y).sum().to('cpu').item()
        tloss /= len(dl)
        tpsnr /= len(dataset)

        # Go back to training mode if the network currently learns.
        if training:
            self.train()
        return {'loss': tloss, 'psnr': tpsnr}

    def fit(self, train_set, *, test_set=None, batch_size: int = 1, epochs: int = 1, verbose: bool = True,
            load_checkpoint_path=None, save_checkpoint_path=None, final_save_path=None,
            save_checkpoint_interval: int = 0, run_idx: str = None, description=None, old_history: Dict = None,
            optimizer_args: dict = {}, scheduler_args: dict(), test_set_validate_interval: int = None,
            tqdm_description: str = None):
        """Optimizes the parameters for the given model based on the given image dataset.
        The dataset should be given by dset.ImageSet.
        That is each element should return a dict with 'clean' and 'noisy' as keys.

        The number of iterations is determined by epochs. Each iteration takes a new subset
        from the training data set. (On the contrary to other approaches were the set is split into
        disjoint subsets.)

        TODO Should be a bit changed as it is too much currently. But this function does its job.
        E.g. Remove save_checkpoint_path, load_checkpoint_path and final_save_path
        as run_idx does this job.

        Args:
            train_set (ImageSet): The training data.
            test_set (_type_, optional): The test data. If given the results are validated onto
                this set. Defaults to None.
            batch_size (int, optional): Determines how big each batch should be.
                It is reduced to the training data set if it is bigger. Defaults to 1.
            epochs (int, optional): The number of iterations. Defaults to 1.
            verbose (bool, optional): Determines if the tqdm-progress bar should be displayed.
                Defaults to True.
            load_checkpoint_path (Path, optional): Path from where the checkpoint should be loaded.
                Defaults to None.
            save_checkpoint_path (Path, optional): Path where the checkpoint should be saved.
                Defaults to None.
            final_save_path (Path, optional): Where the trained model should be saved.
                Defaults to None.
            save_checkpoint_interval (int, optional): Interval when the checkpoints should be saved.
                Defaults to 0.
            run_idx (str, optional): An id of this training run. Defaults to None.
            description (str, optional): A description of this run saved in a text file.
                Defaults to None.
            old_history (Dict, optional): If you want to add an old history
                when e.g. the model was trained before but it was not saved in a checkpoint.
                Defaults to None.
            optimizer_args (dict, optional): Arguments passed to the optimizer (always Adam). Defaults to {}.
            scheduler_args (dict): Arguments passed to the scheduler (always CosineAnnealingLR)
            test_set_validate_interval (int, optional): Number of iterations when
                the performance on the test set should be checked. D
                If not given it is calculated as len(train_set)//batch_size
            tqdm_description (str, optional): Description of the tqdm progress bar. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            The history. A dictionary containing some information about the training run.
        """

        # Prepare the DataLoader
        batch_size = min(batch_size, len(train_set))
        sampler = RandomSampler(train_set, num_samples=batch_size)
        test_set_validate_interval = len(
            train_set)//batch_size if test_set_validate_interval is None else test_set_validate_interval
        # Prepare the optimizer. If path is given load the data.
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **scheduler_args)

        # Prepare all the documentation etc. stuff
        history = dict(
            loss=torch.zeros(epochs),
            psnr=torch.zeros(epochs),
            test_loss=torch.zeros(
                math.ceil(epochs/test_set_validate_interval)),
            test_psnr=torch.zeros(math.ceil(epochs/test_set_validate_interval))
        )
        curr_epoch = 0
        tpsnrmax = 0.
        make_checkpoints_during_run = save_checkpoint_interval > 0 and save_checkpoint_path is not None
        if (save_checkpoint_interval > 0 and save_checkpoint_path is None) or (save_checkpoint_interval <= 0 and save_checkpoint_path is not None):
            raise ValueError(
                "save_checkpoint_path and save_checkpoint_interval should jointly determine that checkpoints are saved.")
        if old_history is not None:
            # If an old history is given append it from the beginning.
            history['loss'] = torch.hstack(
                (old_history['loss'], history['loss']))
            history['psnr'] = torch.hstack(
                (old_history['psnr'], history['psnr']))
            history['test_loss'] = torch.hstack(
                (old_history['test_loss'], history['test_loss']))
            history['test_psnr'] = torch.hstack(
                (old_history['test_psnr'], history['test_psnr']))
            if 'run_idx' in old_history:
                history['run_idx'] = old_history['run_idx']

            # Update the epoch numbers so that they match the history length
            curr_epoch += len(old_history['loss'])
            epochs += len(old_history['loss'])
        if load_checkpoint_path is not None:
            print(
                f'Loading checkpoint from "{load_checkpoint_path}".', end=" ")
            ds = self.load_checkpoint(
                load_checkpoint_path, optimizer=optimizer, scheduler=scheduler, history=history)
            curr_epoch = ds['current_epoch']

            if curr_epoch > epochs:
                print(
                    f"Your choice of epochs = {epochs} is smaller then already learned epochs = {ds['current_epoch']}. Training stopped.")
                return history
            print(f"Starting from epoch={curr_epoch}")
            if make_checkpoints_during_run and os.path.exists(os.path.join(load_checkpoint_path, 'best_test_score.pth')):
                best_score_dict = torch.load(os.path.join(
                    load_checkpoint_path, 'best_test_score.pth'))
                tpsnrmax = best_score_dict['test_psnr'].max().item()

        # Prepare checkpoints during run.
        # If not save_checkpoint_path is given then no checkpoints are saved.
        if make_checkpoints_during_run:
            if run_idx is None:
                if 'run_idx' in history:
                    run_idx = history['run_idx']
                else:
                    now = datetime.now()
                    h, d, mo, ye = now.hour, now.day, now.month, now.year
                    run_idx = f"{ye}_{mo}_{d}_{h}"
                    history['run_idx'] = run_idx
            test_checkpoint_path = os.path.join(
                save_checkpoint_path, str(run_idx), 'best_test_score.pth')
            self.checkpoint_organizer(save_checkpoint_path, run_idx, current_epoch=0, optimizer=optimizer,
                                      scheduler=scheduler, history=history, description=description)

        print(
            f"Start ADAM-training on {self.name}.\n{epochs=}, {len(train_set)=}, {batch_size=}.")

        self.train(mode=True)
        # Set up the text output for the iteration.
        trange = tqdm.trange(curr_epoch, epochs, disable=not verbose,
                             desc=tqdm_description if tqdm_description is not None else self.name)
        infos = dict(
            loss=torch.inf,
            psnr=0.,
            highest=tpsnrmax,
        )

        for epoch in trange:
            sloss = 0.
            spsnr = 0.
            # dnorm = 0.
            # i = 0
            batch = train_set[list(sampler)]
            x = batch['noisy'].to(self.device)
            y = batch['clean'].to(self.device)

            optimizer.zero_grad()

            # Compute prediction error
            pred = self(x)
            loss = loss_func(pred, y)

            # Build the derivative
            loss.backward()

            # Perform a gradient step
            optimizer.step()

            sloss += loss.item()
            with torch.no_grad():
                # pred = self(x)
                spsnr += psnr(pred.detach(), y).mean().to('cpu').item()
                # for p in self.parameters():
                #     dnorm += p.detach().abs().sum().item()
            scheduler.step()

            if (epoch % test_set_validate_interval == 0) or epoch == epochs-1:
                if test_set is not None and len(test_set) > 0:
                    tdic = self.validate(test_set, loss_func)
                    infos['test'] = tdic['psnr']
                    idx = epoch//test_set_validate_interval if epoch != epochs-1 else -1
                    history['test_loss'][idx] = tdic['loss']
                    history['test_psnr'][idx] = tdic['psnr']
                    self.train(True)
                    if tpsnrmax < tdic['psnr']:
                        tpsnrmax = tdic['psnr']
                        infos['highest'] = tpsnrmax
                        if make_checkpoints_during_run:
                            self.save_checkpoint(test_checkpoint_path, history)
            infos['loss'] = sloss  # Mean loss
            infos['psnr'] = spsnr
            # infos['lr'] = optimizer.param_groups[0]['lr']
            # infos['dnorm'] = math.sqrt(dnorm)
            trange.set_postfix(infos)

            history['loss'][epoch] = sloss
            history['psnr'][epoch] = spsnr

            if make_checkpoints_during_run and ((epoch % save_checkpoint_interval) == save_checkpoint_interval - 1):
                self.checkpoint_organizer(save_checkpoint_path, run_idx, current_epoch=epoch, optimizer=optimizer,
                                          scheduler=scheduler, history=history, description=description)

        self.eval()

        if final_save_path is not None:
            if final_save_path == "auto":
                if save_checkpoint_path is not None and save_checkpoint_interval > 0:
                    final_save_path = os.path.join(
                        save_checkpoint_path, str(run_idx), 'final_result.pth')
                else:
                    now = datetime.now()
                    h, m, d, mo, ye = now.hour, now.minute, now.day, now.month, now.year
                    final_save_path = os.path.join(
                        'curious_networks', f'{self.name}_{ye}-{mo}-{d}--{h:02}_{m:02}.pth')

            print(f"Saving the trained model to \"{final_save_path}\"...")
            self.save_checkpoint(final_save_path, history)

        return history

    def conv_fit(self, train_size, test_size, fit_args,
                 transform=None, path_train=None, path_test=None, preload=True,
                 transform_args={},
                 test_transform=None, train_transform=None,
                 ):
        """Conviently sets up a training and test set and calls the fit-method with it.

        It uses the PATH_TRAIN and PATH_TEST path defined on the start of this module.

            train_size (int): size of the training set.
            test_size (int): size of the test set.
            fit_args (dict): Arguments which are passed to the fit function.
            transform (_type_, optional): Transformation to use on the (clean) input images.
                Defaults to the standard transform of dset-module.
            path_train (path, optional): If you do not want to use the default set.
                Defaults to the train set of BSR-set.
            path_test (path, optional): If you do not want to use the default set.
                Defaults to the test set of BSR-set.
            preload (bool, optional): Determines if the data should be preloaded.
                This can speed up the training dramatically. Defaults to True.
            transform_args (dict, optional): Arugments passed to the (standard) transformation. Defaults to {}.
            test_transform (Callable, optional): A transformation which is only used on the test set.
                Defaults to None.
            train_transform (Callable, optional): A transformation which is only used on the training set.
                Defaults to None.

        Returns:
            The history from the fit function and the arguments passed to it.
        """

        fit_args = {**fit_args}
        if 'train_set' not in fit_args and 'test_set' not in fit_args:
            if path_train is None:
                path_train =PATH_TRAIN
            if path_test is None:
                path_test = PATH_TEST
            if transform is None:
                transform = dset.StandardTransform(**transform_args)
            if test_transform is None:
                test_transform = transform
            if train_transform is None:
                train_transform = transform
            train_set, test_set = dset.getTrainTestSet(
                path_train=path_train,
                path_test=path_test,
                max_train_size=train_size,
                max_test_size=test_size,
                test_transform=test_transform,
                train_transform=train_transform,
                preload=preload,
            )
            fit_args['train_set'] = train_set
            fit_args['test_set'] = test_set
        return self.fit(**fit_args), fit_args


class Prox_l1(nn.Module):
    """prox_*l1 operator.
    For the case: ||x||_1
    """

    def __init__(self, sigma=1.) -> None:
        super().__init__()
        self.sigma = sigma
        self.t1 = torch.tensor(sigma, dtype=torch.float)

    def forward(self, xk):
        return torch.maximum(torch.minimum(self.t1, xk), -self.t1)

    def __repr__(self):
        return f"Prox_l1(sigma={self.t1.item()})"


class Prox_l1_f(nn.Module):
    """prox_l1 operator.
    For the case: ||x-f||_1
    """

    def __init__(self, sigma=1.) -> None:
        super().__init__()
        self.sigma = sigma
        self.t1 = torch.tensor(sigma, dtype=torch.float)

    def forward(self, xk, f):
        return torch.maximum(torch.tensor(0.), torch.abs(xk - f) - self.t1)*torch.sign(xk - f) + f

    def __repr__(self):
        return f"Prox_l1(sigma={self.t1.item()})"


class Prox_l1_test(nn.Module):
    """Testing"""

    def __init__(self, sigma=1.) -> None:
        super().__init__()
        self.sigma = sigma
        self.t1 = torch.tensor(sigma, dtype=torch.float)

    def forward(self, xk):
        # return torch.maximum(torch.minimum(self.t1, xk), -self.t1)
        return xk - torch.maximum(torch.tensor(0.), torch.abs(xk) - self.t1)*torch.sign(xk)

    def __repr__(self):
        return f"Prox_l1(sigma={self.t1.item()})"


class Prox_l2(nn.Module):
    """prox_l2 operator.
    For the case: ||x-f||^2_2
    """

    def __init__(self, tau=0.1) -> None:
        super().__init__()
        self.t1 = torch.tensor(1., dtype=torch.float)
        # self.tau = torch.tensor(tau, dtype=torch.float)
        self.tau = torch.tensor(tau)

    def forward(self, xk, f):
        return (self.tau*f + xk)/(self.t1 + self.tau)

    def __repr__(self):
        return f"Prox_l12(tau={self.tau.item()})"


class Prox_l2_learnable(nn.Module):
    """Learnable prox_l2 operator, the parameter tau can be changed by the optimizer.
    For the case: ||x-f||^2_2
    """

    def __init__(self, tau=0.01) -> None:
        super().__init__()
        self.t1 = torch.tensor(1., dtype=torch.float)
        # self.tau = torch.tensor(tau, dtype=torch.float)
        self.tau = torch.nn.Parameter(torch.tensor(tau), requires_grad=True)

    def forward(self, xk, f):
        """Application of this proximal operator onto the input xk. f is the original noisy image.
        """
        return (self.tau*f + xk)/(self.t1 + self.tau)

    def __repr__(self):
        return f"Prox_l12(tau={self.tau.item()})"


class Prox_l1l2_f(nn.Module):
    """prox_l1l2 operator.
    For the model: lam1*||x-f||_1 + lam2/2*||x-f||^2_2"""

    def __init__(self, lam1, lam2, tau) -> None:
        super().__init__()
        self.lam1 = torch.tensor(lam1, dtype=torch.float)
        self.lam2 = torch.tensor(lam2, dtype=torch.float)
        self.tau = torch.tensor(tau, dtype=torch.float)

    def forward(self, xk, f):
        """Application of this proximal operator onto the input xk. f is the original noisy image.
        """
        factor = 1/(self.tau*self.lam2 + 1)
        return factor*torch.maximum(torch.tensor(0.), torch.abs(xk - f) - self.lam1*self.tau)*torch.sign(xk - f) + f


class Prox_l2T(nn.Module):
    """prox_l2* operator. Proximal operator of the conjugate of ||x-f||^2_2"""

    def __init__(self, sigma) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, xk, f):
        """Application of this proximal operator onto the input xk. f is the original noisy image.
        """
        return (xk - self.sigma*f)/(1+self.sigma)


class Prox_l2T_learnable(nn.Module):
    """prox_l2* operator. Proximal operator of the conjugate of ||x-f||^2_2.
    The operator sigma is learnable."""

    def __init__(self, sigma) -> None:
        super().__init__()
        self.sigma = torch.nn.Parameter(torch.tensor(
            sigma, dtype=torch.float32), requires_grad=True)

    def forward(self, xk, f):
        """Application of this proximal operator onto the input xk. f is the original noisy image.
        """
        return (xk - self.sigma*f)/(1+self.sigma)


class Prox_indicator(nn.Module):
    """The proximal operator of an indicator function."""

    def forward(self, x, f, xset):
        """Application of this proximal operator onto the input xk. f is the original noisy image.
        xset should be a mask with the set of the indicator function.
        """
        xc = x.clone()
        xc[~xset] = f[~xset]
        return xc
