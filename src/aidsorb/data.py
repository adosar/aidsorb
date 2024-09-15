# This file is part of AIdsorb.
# Copyright (C) 2024 Antonios P. Sarikas

# AIdsorb is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

r"""
This module provides helper functions and classes for creating datasets and
handling point clouds of variable sizes.
"""

import os
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
from . _internal import _SEED, pd


def prepare_data(source: str, split_ratio: Sequence=(0.8, 0.1, 0.1), seed: int=_SEED):
    r"""
    Split a source of point clouds in train, validation and test sets.

    Each ``.json`` file that is created, stores the names of the point clouds
    that will be used for *training*, *validation* and *testing*.

    .. warning::
        * No directory is created by :func:`prepare_data`. All ``.json`` files
          are stored under the directory containing ``source``.
        * Splitting doesn't support stratification. If your dataset is small and
          you want to perform classification, consider using
          `train_test_split`_.

    .. _train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    Parameters
    ----------
    source : str
        Absolute or relative path to the file holding the point clouds.
    split_ratio : sequence, default=(0.8, 0.1, 0.1)
        The sizes or fractions of splits to be produced.

        * ``split_ratio[0] == train``.
        * ``split_ratio[1] == validation``.
        * ``split_ratio[2] == test``.

    seed : int, default=1
        Controls the randomness of the ``rng`` used for splitting.

    Examples
    --------
    Before the split::

        pcd_data
        └──source.npz

    >>> prepare_data('path/to/pcd_data/source.npz')  # doctest: +SKIP

    After the split::

        pcd_data
        ├──source.npz
        ├──train.json
        ├──validation.json
        └──test.json
    """
    rng = torch.Generator().manual_seed(seed)
    path = Path(source).parent
    pcds = np.load(source)

    train, val, test = random_split(pcds.files, split_ratio, generator=rng)

    for split, mode in zip((train, val, test), ('train', 'validation', 'test')):
        names = list(split)
        with open(os.path.join(path, f'{mode}.json'), 'w') as fhand:
            json.dump(names, fhand, indent=4)

    print('\033[32mData preparation completed!\033[0m')


def get_names(filename):
    r"""
    Return names stored in a ``.json`` file.

    Parameters
    ----------
    filename : str
        The name of the file from which names will be retrieved.

    Returns
    -------
    names : list
    """
    with open(filename, 'r') as fhand:
        names = json.load(fhand)

    return names


def upsample_pcd(pcd, size):
    r"""
    Upsample ``pcd`` to a new ``size`` by sampling with replacement from ``pcd``.

    Parameters
    ----------
    pcd : tensor of shape (N, C)
        The original point cloud of size ``N``.
    size : int
        The size of the new point cloud.

    Returns
    -------
    new_pcd : tensor of shape (size, C).

    Examples
    --------
    >>> pcd = torch.tensor([[2, 4, 5, 6]])
    >>> upsample_pcd(pcd, 3)
    tensor([[2, 4, 5, 6],
            [2, 4, 5, 6],
            [2, 4, 5, 6]])

    >>> # New points point must be from pcd.
    >>> pcd = torch.randn(10, 4)
    >>> new_pcd = upsample_pcd(pcd, 20)
    >>> (new_pcd[-1] == pcd).all(1).any()  # Check for last point.
    tensor(True)

    >>> # No upsampling.
    >>> pcd = torch.randn(100, 4)
    >>> new_pcd = upsample_pcd(pcd, len(pcd))
    >>> torch.equal(pcd, new_pcd)
    True

    """
    n_samples = size - len(pcd)
    indices = torch.from_numpy(np.random.choice(len(pcd), n_samples, replace=True))
    new_points = pcd[indices]

    return torch.cat((pcd, new_points))


def pad_pcds(pcds, channels_first=True, mode='upsample'):
    r"""
    Pad a sequence of variable size point clouds.

    Each point cloud must have shape ``(N_i, C)``.

    Parameters
    ----------
    pcds : sequence of tensors
    mode : {'zeropad', 'upsample'}, default='upsample'
    channels_first : bool, default=True

    Returns
    -------
    batch : tensor of shape (B, T, C) or (B, C, T)
         If ``channels_first=False``, then ``batch`` has shape ``(B, T, C)``,
         where  ``B == len(pcds)`` is the batch size and ``T`` is the size of
         the largest point cloud in ``pcds``. Otherwise, ``(B, C, T)``.
         
    See Also
    --------
    :func:`upsample_pcd` : For a description of ``'upsample'`` mode.
    :func:`torch.nn.utils.rnn.pad_sequence` : For a description of ``'zeropad'`` mode.

    Examples
    --------
    >>> x1 = torch.tensor([[1, 2, 3, 4]])
    >>> x2 = torch.tensor([[2, 5, 3, 8], [0, 2, 8, 9]])

    >>> batch = pad_pcds((x1, x2), channels_first=False)
    >>> batch
    tensor([[[1, 2, 3, 4],
             [1, 2, 3, 4]],
    <BLANKLINE>
            [[2, 5, 3, 8],
             [0, 2, 8, 9]]])

    >>> batch = pad_pcds((x1, x2), channels_first=True)
    >>> batch
    tensor([[[1, 1],
             [2, 2],
             [3, 3],
             [4, 4]],
    <BLANKLINE>
            [[2, 0],
             [5, 2],
             [3, 8],
             [8, 9]]])

    >>> batch = pad_pcds((x1, x2), channels_first=False, mode='zeropad')
    >>> batch
    tensor([[[1, 2, 3, 4],
             [0, 0, 0, 0]],
    <BLANKLINE>
            [[2, 5, 3, 8],
             [0, 2, 8, 9]]])

    >>> batch = pad_pcds((x1, x2), channels_first=True, mode='zeropad')
    >>> batch
    tensor([[[1, 0],
             [2, 0],
             [3, 0],
             [4, 0]],
    <BLANKLINE>
            [[2, 0],
             [5, 2],
             [3, 8],
             [8, 9]]])
    """
    if mode == 'zeropad':
        batch = pad_sequence(pcds, batch_first=True, padding_value=0)

    elif mode == 'upsample':
        max_len = max(len(i) for i in pcds)
        new_pcds = [upsample_pcd(p, max_len) if len(p) < max_len else p for p in pcds]
        batch = torch.stack(new_pcds)

    # Shape (B, n_points, C).
    if channels_first:
        batch = batch.transpose(1, 2)  # Shape (B, C, n_points).

    return batch


class Collator():
    r"""
    Collate a sequence of samples into a ``batch``.

    Point clouds are padded before collation, so they can form a batch.

    .. rubric:: Shapes

    * Input: sequence of samples

        Each sample is a tuple of tensors ``(pcd, label)``, where
        ``pcd`` has shape ``(N_i, C)`` and ``label`` has shape
        ``(n_outputs,)`` or ``()``.

    * Output: tuple of length 2

        * ``batch[0] == x`` with shape ``(B, C, T)`` if ``channels_first=True``,
          otherwise ``(B, T, C)``. ``B`` is the batch size and ``T`` is the size
          of the largest point cloud in the sequence.
        * ``batch[1] == y`` with shape ``(B, n_outputs)`` or ``(B,)``.

    .. tip::
        Use an instance of this class as ``collate_fn`` with
        ``channels_first=True``, if your model is :class:`~aidsorb.models.PointNet`.

    .. todo::
        Add functionality for collating only point clouds (useful when the
        dataset is unlabeled).

    Parameters
    ----------
    channels_first : bool, default=True
    mode : {'zeropad', 'upsample'}, default='upsample'

    See Also
    --------
    :func:`pad_pcds` : For a description of the parameters.
    :func:`upsample_pcd` : For a description of the parameters.

    Examples
    --------
    >>> sample1 = (torch.tensor([[1, 4, 5, 2]]), torch.tensor([1., 2.]))
    >>> sample2 = (torch.tensor([[0, 4, 0, 2], [2, 4, 1, 8]]), torch.tensor([7., 3.]))

    >>> collate_fn = Collator()
    >>> x, y = collate_fn((sample1, sample2))
    >>> x.shape
    torch.Size([2, 4, 2])
    >>> y.shape
    torch.Size([2, 2])
    >>> x
    tensor([[[1, 1],
             [4, 4],
             [5, 5],
             [2, 2]],
    <BLANKLINE>
            [[0, 2],
             [4, 4],
             [0, 1],
             [2, 8]]])
    >>> y
    tensor([[1., 2.],
            [7., 3.]])

    >>> collate_fn = Collator(channels_first=False, mode='zeropad')
    >>> x, y = collate_fn((sample1, sample2))
    >>> x
    tensor([[[1, 4, 5, 2],
             [0, 0, 0, 0]],
    <BLANKLINE>
            [[0, 4, 0, 2],
             [2, 4, 1, 8]]])
    >>> y
    tensor([[1., 2.],
            [7., 3.]])

    >>> # Label has shape (), i.e. is scalar.
    >>> sample1 = (torch.tensor([[3, 4, 3, 2]]), torch.tensor(0))
    >>> sample2 = (torch.tensor([[2, 4, 8, 2], [9, 4, 1, 8]]), torch.tensor(1))
    >>> collate_fn = Collator(channels_first=False, mode='zeropad')
    >>> x, y = collate_fn((sample1, sample2))
    >>> x
    tensor([[[3, 4, 3, 2],
             [0, 0, 0, 0]],
    <BLANKLINE>
            [[2, 4, 8, 2],
             [9, 4, 1, 8]]])
    >>> y
    tensor([0, 1])
    """
    def __init__(self, channels_first=True, mode='upsample'):
        self.channels_first = channels_first
        self.mode = mode

    def __call__(self, samples):
        r"""
        Parameters
        ----------
        samples : sequence of tuples
            Each sample is a tuple of tensors ``(pcd, label)`` where
            ``pcd.shape == (n_points, C)`` and ``label`` has shape
            ``(n_outputs,)`` or ``()``.

        Returns
        -------
        batch : tuple of length 2
            * ``batch[0] == x`` with shape ``(B, C, T)`` or ``(B, T, C)``, where
              ``T`` is the size of the largest point cloud.
            * ``batch[1] == y`` with shape ``(B, n_outputs)`` or ``(B,)``.
        """
        pcds, labels = list(zip(*samples))
        
        x = pad_pcds(pcds, channels_first=self.channels_first, mode=self.mode)
        y = torch.stack(labels)

        return x, y


class PCDDataset(Dataset):
    r"""
    ``Dataset`` for point clouds.

    .. tip::
        For implementing your own transforms, have a look at the transforms
        `tutorial`_.  For more flexibility, consider implementing them as
        callable instances of classes.

    .. _tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms

    Parameters
    ----------
    pcd_names : list
        List containing the names of the point clouds.
    path_to_X : str
        Absolute or relative path to the ``.npz`` file holding the point clouds.
    path_to_Y : str, optional
        Absolute or relative path to the ``.csv`` file holding the labels of the
        point clouds.

        .. warning::
            The comma ``,`` is assumed as the field separator.

    index_col : str, optional
        Column name of the ``.csv`` file to be used as row labels. The names
        (values) under this column must follow the same naming scheme as in
        ``pcd_names``.
    labels : list, optional
        List containing the names of the properties to be predicted. No effect
        if ``path_to_Y=None``.
    transform_x : callable, optional
        Transforms applied to ``input``, i.e to each point cloud.
    transform_y : callable, optional
        Transforms applied to ``output``. No effect if ``path_to_Y=None``.

    See Also
    --------
    :mod:`aidsorb.transforms` : For available point cloud transformations.
    """
    def __init__(
            self, pcd_names, path_to_X,
            path_to_Y=None, index_col=None, labels=None,
            transform_x=None, transform_y=None,
            ):

        if (labels is not None) and (type(labels) != list):
            raise ValueError('labels must be a list!')

        self._pcd_names = pcd_names
        self.path_to_X = path_to_X
        self.path_to_Y = path_to_Y
        self.labels = labels
        self.index_col = index_col
        self.transform_x = transform_x
        self.transform_y = transform_y

        self.X = None
        self.Y = None

    @property
    def pcd_names(self):
        r"""The names of the point clouds."""
        return self._pcd_names

    def __len__(self):
        return len(self.pcd_names)

    def __getitem__(self, idx):
        # Account for np.load and multiprocessing.
        if self.X is None:
            self.X = np.load(self.path_to_X)
        if self.Y is None and self.path_to_Y is not None:
            self.Y = pd.read_csv(
                    self.path_to_Y,
                    index_col=self.index_col,
                    usecols=[*self.labels, self.index_col],
                    )

        name = self.pcd_names[idx]
        sample_x = self.X[name]

        if self.transform_x is not None:
            sample_x = self.transform_x(sample_x)

        # Only for labeled datasets.
        if self.Y is not None:
            sample_y = self.Y.loc[name].to_numpy()

            if self.transform_y is not None:
                sample_y = self.transform_y(sample_y)

            return (
                    torch.tensor(sample_x, dtype=torch.float),
                    torch.tensor(sample_y, dtype=torch.float)
                    )

        return torch.tensor(sample_x, dtype=torch.float)
