r"""
Add docstring of the module.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
from . _internal import _SEED


def prepare_data(source, split_ratio=(0.8, 0.1, 0.1), seed=_SEED):
    r"""
    Split a source of point clouds in train, validation and test sets.

    Before the split::

        pcd_data
        └──source.npz

    After the split::

        pcd_data
        ├──source.npz
        ├──train.json
        ├──validation.json
        └──test.json

    Each ``.json`` file stores the names of the point clouds.

    .. warning::
        No directory is created by :func:`prepare_data`. **All ``.json`` files
        are stored under the directory containing ``source``**.

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
    """
    rng = torch.Generator().manual_seed(seed)
    path = Path(source).parent
    pcds = np.load(source, mmap_mode='r')

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
    pcd : array of shape (N, *).
        The original point cloud of size ``N``.
    size : int
        The size of the new point cloud.

    Returns
    -------
    new_pcd : array of shape (size, *).

    Examples
    --------
    >>> pcd = torch.tensor([[2, 4, 5, 6]])
    >>> upsample_pcd(pcd, 3)
    tensor([[2, 4, 5, 6],
            [2, 4, 5, 6],
            [2, 4, 5, 6]])
    >>> upsample_pcd(pcd, 1)  # No upsampling.
    tensor([[2, 4, 5, 6]])
    """
    n_samples = size - len(pcd)
    indices = torch.from_numpy(np.random.choice(len(pcd), n_samples))
    new_points = pcd[indices]

    return torch.cat((pcd, new_points))


def pad_pcds(pcds, channels_first=True, mode='upsample'):
    r"""
    Pad a sequence of variable size point clouds.

    Each point cloud must have shape ``(N, *)``.

    Parameters
    ----------
    pcds : sequence of tensors
    mode : {'zeropad', 'upsample'}, default='upsample'
    channels_first : bool, default=True

    Returns
    -------
    batch : tensor of shape (B, T, *) or (B, *, T)
        ``B == len(pcds)`` is the batch size and ``T`` is the size of the
        largest point cloud in ``pcds``. If ``channels_first`` is ``False``, the
        batch has shape ``(B, T, *)``. Otherwise, ``(B, *, T)``.

    See Also
    --------
    :func:`usample_pcd` : For a description of ``'upsample'`` mode.
    `pad_sequence`_ : For a description of ``'zeropad'`` mode.

    Examples
    --------
    >>> x1 = torch.tensor([[1, 2, 3, 4]])
    >>> x2 = torch.tensor([[2, 5, 3, 8], [0, 2, 8, 9]])
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

    .. _pad_sequence: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
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
    Collate point clouds and labels.

    Point clouds are padded before collation. See :func:`pad_pcds`.

    .. note::
        You should use an instance of this class as ``collate_fn`` if your model
        is :class:`models.PointNet`.

    Parameters
    ----------
    channels_first : bool, default=True
    mode : {'zeropad', 'sample'}, default='upsample'

    See Also
    --------
    :func:`pad_pcds`

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
            ``pcd.shape == (n_points, *)`` and ``label.shape == (n_outputs,)``.

        Returns
        -------
        batch : tuple of shape (2,)
            * ``x == batch[0]`` with shape ``(B, *, T)``, where ``T`` is the size of
            the largest point cloud.
            * ``y == batch[1]`` with shape ``(B, n_outputs)``.
        """
        pcds, labels = list(zip(*samples))
        
        x = pad_pcds(pcds, channels_first=self.channels_first, mode=self.mode)
        y = torch.stack(labels)  # Shape (B, n_outputs).

        return x, y


class PCDDataset(Dataset):
    r"""
    Dataset for point clouds.

    Parameters
    ----------
    pcd_names : list
        List containing the names of the point clouds.
    path_to_X : str
        Absolute or relative path to the ``.npz`` file holding the point clouds.
    path_to_Y : str, optional
        Absolute or relative path to the ``.csv`` file holding the labels of the
        point clouds.
    index_col : str, optional
        Column name of the ``.csv`` file to be used as row labels.
    labels : list, optional
        List containing the names of the properties to be predicted. No effect
        if ``path_to_Y == None``.
    transform_x : callable, optional
        Transforms applied to ``sample_x`` (i.e to each point cloud). See
        `transforms`_ for implementing your own transforms.
    transform_y : callable, optional
        Transforms applied to ``sample_y`` (i.e. to the individual outputs).
        See `transforms`_ for implementing your own transforms. No effect if
        ``pcd_Y == None``.

        .. note::
            For example, if you want to perform classification, here you can
            pass the one-hot encoder (if the dataset is not already preprocessed).

    .. _transforms: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
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
        return self._pcd_names

    def __len__(self):
        return len(self.pcd_names)

    def __getitem__(self, idx):
        # Account for np.load and multiprocessing.
        if self.X is None:
            self.X = np.load(self.path_to_X, mmap_mode='r')
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
            sample_y = self.Y.loc[name].values

            if self.transform_y is not None:
                sample_y = self.transform_y(sample_y)

            return (
                    torch.tensor(sample_x, dtype=torch.float),
                    torch.tensor(sample_y, dtype=torch.float)
                    )

        return torch.tensor(sample_x, dtype=torch.float)
