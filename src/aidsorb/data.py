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
Helper functions and classes for creating datasets and handling point clouds of
variable sizes.
"""

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split

from ._internal import pd
from .transforms import upsample_pcd


def prepare_data(
        source: str,
        split_ratio: Sequence | None = None,
        seed: int = 1,
        ) -> None:
    r"""
    Split point clouds into train, validation and test sets.

    Each ``.json`` file that is created, stores the names of the point clouds
    that will be used for training, validation and testing.

    .. warning::
        * All ``.json`` files are stored under the parent directory of ``source``.
        * Splitting doesn't support stratification. If your dataset is small and
          you want to perform classification, consider using
          `train_test_split`_.

    .. _train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    Parameters
    ----------
    source : str
        Absolute or relative path to the directory holding the point clouds.
    split_ratio : sequence, default=None
        Absolute sizes or fractions of splits of the form ``(train, val,
        test)``. If :obj:`None`, it is set to ``(0.8, 0.1, 0.1)``.

    seed : int, default=1
        Controls randomness of the ``rng`` used for splitting.

    Examples
    --------
    Before the split::

        project_root
        └── source
            ├── foo.npy
            ├── ...
            └── bar.npy

    >>> prepare_data('path/to/source')  # doctest: +SKIP

    After the split::

        project_root
        ├── source
        │   ├── foo.npy
        │   ├── ...
        │   └── bar.npy
        ├── test.json
        ├── train.json
        └── validation.json
    """
    rng = torch.Generator().manual_seed(seed)
    path = Path(source).parent
    pcd_names = [name.removesuffix('.npy') for name in sorted(os.listdir(source))]

    # Set default split ratio.
    if split_ratio is None:
        split_ratio = (0.8, 0.1, 0.1)

    # Split the names of the point clouds.
    train, val, test = random_split(pcd_names, split_ratio, generator=rng)

    for split, mode in zip((train, val, test), ('train', 'validation', 'test')):
        names = list(split)
        filename = os.path.join(path, f'{mode}.json')

        with open(filename, 'w') as fhand:
            json.dump(names, fhand, indent=4)

        print(f'Created file: \033[0;34m{filename}\033[0m')

    print('\033[32;1mData preparation completed!\033[0m')


def get_names(filename: str) -> tuple:
    r"""
    Return point cloud names stored in a ``.json`` file.

    Parameters
    ----------
    filename : str
        Absolute or relative path to the file.

    Returns
    -------
    names : tuple
    """
    with open(filename, 'r') as fhand:
        return tuple(json.load(fhand))


def pad_pcds(
        pcds: Sequence[Tensor],
        *,
        channels_first: bool,
        mode: str = 'upsample',
        return_mask: bool = False,
        ) -> Tensor | tuple:
    r"""
    Pad a sequence of variable size point clouds.

    Each point cloud must have shape ``(N_i, C)``.

    .. rubric:: Shapes

    * ``batch`` tensor of shape ``(B, T, C)`` if ``channels_first=False``,
      else ``(B, C, T)``.
    * ``mask`` boolean tensor of shape ``(B, T)`` where :obj:`True` indicates
      padding.

    ``B`` is the batch size and ``T`` is the size of the largest point
    cloud in the sequence.

    Parameters
    ----------
    pcds : sequence of tensors
    channels_first : bool
    mode : {'zeropad', 'upsample'}, default='upsample'
    return_mask : bool, default=False

    Returns
    -------
    tensor or tuple of tensors
        ``batch`` if ``return_mask=False``, else ``(batch, mask)``.

    See Also
    --------
    :func:`~.upsample_pcd` : For a description of ``'upsample'`` mode.
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

    >>> # Pad and return padding mask (useful for attention-based architectures).
    >>> batch, mask = pad_pcds((x1, x2), channels_first=False, return_mask=True)
    >>> batch
    tensor([[[1, 2, 3, 4],
             [1, 2, 3, 4]],
    <BLANKLINE>
            [[2, 5, 3, 8],
             [0, 2, 8, 9]]])
    >>> mask
    tensor([[False,  True],
            [False, False]])

    >>> # Pad a single point cloud.
    >>> pad_pcds([x1], channels_first=False, mode='zeropad')
    tensor([[[1, 2, 3, 4]]])
    >>> pad_pcds([x1], channels_first=True, mode='upsample')
    tensor([[[1],
             [2],
             [3],
             [4]]])
    """
    pcd_len = torch.tensor([len(p) for p in pcds])
    max_len = pcd_len.max().item()

    if mode == 'zeropad':
        batch = pad_sequence(
                pcds, batch_first=True,
                padding_value=0.0, padding_side='right'
                )
    elif mode == 'upsample':
        padded_pcds = [upsample_pcd(p, max_len) if len(p) < max_len else p for p in pcds]
        batch = torch.stack(padded_pcds)  # Shape (B, max_len, C).

    if channels_first:
        batch = batch.transpose(1, 2)  # Shape (B, C, max_len).

    # Note: right padding is assumed.
    if return_mask:
        mask = torch.arange(max_len)[None] >= pcd_len[:, None]
        return batch, mask

    return batch


class Collator:
    r"""
    Collate a sequence of samples into a batch.

    Point clouds are padded before collation, so they can form a batch.

    .. rubric:: Shapes

    * Input: sequence of samples

        Each sample is a tuple of ``(pcd, label)``.

        * ``pcd`` tensor of shape ``(N_i, C)``.
        * ``label`` tensor of shape ``(n_outputs,)``, ``()`` or :obj:`None`.

    * Output: tuple

        If ``return_mask=False``, then output is ``(x, y)``, else ``((x, mask), y)``.

        * ``x`` tensor of shape ``(B, C, T)`` if ``channels_first=True``, else ``(B, T, C)``.
        * ``y`` tensor of  shape ``(B, n_outputs)``, ``(B,)`` or :obj:`None`.
        * ``mask`` boolean tensor of shape ``(B, T)`` where :obj:`True` indicates padding.

     ``B`` is the batch size and ``T`` is the size of the largest point cloud in the
     sequence.

    Parameters
    ----------
    channels_first : bool
    mode : {'zeropad', 'upsample'}, default='upsample'
    return_mask : bool, default=False

    See Also
    --------
    :func:`pad_pcds` : For a description of the parameters.

    Examples
    --------
    >>> sample1 = (torch.tensor([[1, 4, 5, 2]]), torch.tensor([1., 2.]))
    >>> sample2 = (torch.tensor([[0, 4, 0, 2], [2, 4, 1, 8]]), torch.tensor([7., 3.]))

    >>> collate_fn = Collator(channels_first=True)
    >>> x, y = collate_fn((sample1, sample2))
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

    >>> # Label is None, i.e. unlabeled data.
    >>> sample1 = (torch.tensor([[1., 0., 1., 0.]]), None)
    >>> sample2 = (torch.tensor([[5., 2., 2., 0.], [9., 0., 0., 1.]]), None)
    >>> collate_fn = Collator(channels_first=True, mode='zeropad')
    >>> x, y = collate_fn((sample1, sample2))
    >>> x
    tensor([[[1., 0.],
             [0., 0.],
             [1., 0.],
             [0., 0.]],
    <BLANKLINE>
            [[5., 9.],
             [2., 0.],
             [2., 0.],
             [0., 1.]]])
    >>> y

    >>> # Collate and return padding mask.
    >>> sample1 = (torch.tensor([[4, 2, 1, 4], [2, 0, 0, 1]]), torch.tensor(1))
    >>> sample2 = (torch.tensor([[1, 2, 3, 1]]), torch.tensor(4))
    >>> collate_fn = Collator(channels_first=False, mode='zeropad', return_mask=True)
    >>> (x, mask), y = collate_fn((sample1, sample2))
    >>> x
    tensor([[[4, 2, 1, 4],
             [2, 0, 0, 1]],
    <BLANKLINE>
            [[1, 2, 3, 1],
             [0, 0, 0, 0]]])
    >>> y
    tensor([1, 4])
    >>> mask
    tensor([[False, False],
            [False,  True]])

    >>> # Batch a single unlabeled sample.
    >>> sample = (torch.tensor([[2, 3, 4]]), None)
    >>> collate_fn = Collator(channels_first=False)
    >>> x, y = collate_fn([sample])
    >>> x
    tensor([[[2, 3, 4]]])
    >>> y

    >>> # Batch a single labeled sample.
    >>> sample = (torch.tensor([[1, 1, 2]]), torch.tensor(10))
    >>> collate_fn = Collator(channels_first=True, mode='zeropad')
    >>> x, y = collate_fn([sample])
    >>> x
    tensor([[[1],
             [1],
             [2]]])
    >>> y
    tensor([10])
    """
    def __init__(
            self,
            *,
            channels_first: bool,
            mode: str = 'upsample',
            return_mask: bool = False,
            ) -> None:

        self.channels_first = channels_first
        self.mode = mode
        self.return_mask = return_mask

    def __call__(
            self,
            samples: Sequence[tuple[Tensor, Tensor | None]],
            ) -> tuple[Tensor, Tensor | None]:
        r"""
        Parameters
        ----------
        samples : sequence of tuples
            Each sample is a tuple of tensors ``(pcd, label)`` or ``(pcd,
            None)``.

        Returns
        -------
        tuple
            ``(x, y)`` or ``(x, None)``. If ``return_mask=True``, then
            ``x`` is a tuple ``(batch, mask)``, else ``batch``.
        """
        pcds, labels = list(zip(*samples))

        x = pad_pcds(
                pcds, channels_first=self.channels_first,
                mode=self.mode, return_mask=self.return_mask
                )
        y = torch.stack(labels) if None not in labels else None

        return x, y


class PCDDataset(Dataset):
    r"""
    :class:`~torch.utils.data.Dataset` for point clouds.

    Indexing the dataset returns ``(x, None)`` if data are unlabeled, i.e.
    ``path_to_Y=None``, else ``(x, y)``, where ``x`` and ``y`` are the results of
    ``transform_x`` and ``transform_y``, respectively.

    .. note::
        * All data (i.e. point cloud and its label) are converted to
          :class:`~.torch.Tensor` before passed to transforms. As such,
          ``transform_x`` and ``transform_y`` expect :class:`~.torch.Tensor` as
          input.
        * ``y`` has shape ``(len(labels),)`` if ``transform_y=None``.
        * Comma ``,`` is assumed as the field separator in ``.csv`` file.

    Parameters
    ----------
    pcd_names : sequence
        Point cloud names.
    path_to_X : str
        Absolute or relative path to the directory holding the point clouds.
    path_to_Y : str, optional
        Absolute or relative path to the ``.csv`` file holding the labels of the
        point clouds.
    index_col : str, optional
        Column name of the ``.csv`` file to be used for indexing. This column
        must include ``pcd_names``. No effect if ``path_to_Y=None``.
    labels : list, optional
        List of column names from the ``.csv`` file containing the properties to be
        predicted. No effect if ``path_to_Y=None``.
    transform_x : callable, optional
        Transformation to apply to point cloud.
    transform_y : callable, optional
        Transformation to apply to label. No effect if ``path_to_Y=None``.

    See Also
    --------
    :mod:`aidsorb.transforms` : For available point cloud transformations.
    """
    def __init__(
            self,
            pcd_names: Sequence[str],
            path_to_X: str,
            *,
            path_to_Y: str | None = None,
            index_col: str | None = None,
            labels: list[str] | None = None,
            transform_x: Callable | None = None,
            transform_y: Callable | None = None,
            ) -> None:

        self._pcd_names = tuple(pcd_names)  # Immutable for safety.
        self.path_to_X = path_to_X
        self.path_to_Y = path_to_Y
        self.index_col = index_col
        self.labels = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

        #: Dataframe for the labels. The columns follow the order in ``labels``.
        self.Y = None

        if self.path_to_Y is not None:  # Only for labeled datasets.
            self.Y = pd.read_csv(
                    self.path_to_Y,
                    index_col=self.index_col,
                    usecols=[*self.labels, self.index_col],
                    )[self.labels]

    @property
    def pcd_names(self) -> tuple:
        r"""Point cloud names."""
        return self._pcd_names

    def __len__(self) -> int:
        return len(self.pcd_names)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor | None]:
        pcd_name = self.pcd_names[idx]
        pcd_path = os.path.join(self.path_to_X, f'{pcd_name}.npy')

        pcd = torch.tensor(np.load(pcd_path), dtype=torch.float)
        label = None

        if self.transform_x is not None:
            pcd = self.transform_x(pcd)

        if self.Y is not None:
            y_arr = self.Y.loc[pcd_name].to_numpy()
            dtype = torch.float if np.issubdtype(y_arr.dtype, np.floating) else None
            label = torch.tensor(y_arr, dtype=dtype)

            if self.transform_y is not None:
                label = self.transform_y(label)

        return pcd, label
