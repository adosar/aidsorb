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
This module provides :class:`~lightning.pytorch.core.LightningDataModule`'s for
use with |lightning|.
"""

import os
from typing import Callable
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
from . data import get_names, PCDDataset


class PCDDataModule(L.LightningDataModule):
    r"""
    ``LightningDataModule`` for point clouds.

    .. note::
        The following directory structure is assumed::

            pcd_data
            ├──pcds.npz        <-- path_to_X
            ├──train.json
            ├──validation.json
            └──test.json

    .. tip::
        Assuming ``pcd_data/pcds.npz`` already exists, you can create the above
        directory structure with :func:`~aidsorb.data.prepare_data`.

    .. todo::
        * Add support for ``predict_dataloader``.
        * Add option ``drop_last`` for ``train_dataloader``.

    Parameters
    ----------
    path_to_X : str
        Absolute or relative path to the ``.npz`` file holding the point clouds.
    path_to_Y : str
        Absolute or relative path to the ``.csv`` file holding the labels of the
        point clouds.

        .. warning::
            The comma ``,`` is assumed as the field separator.

    index_col : str
        Column name of the ``.csv`` file to be used as row labels. The names
        (values) under this column must follow the same naming scheme as in
        ``pcds.npz``.
    labels : list
        List containing the names of the properties to be predicted. No effect if
        ``path_to_Y=None``.
    train_size : int, optional
        The number of training samples. By default, all training samples are
        used.
    train_transform_x : callable, optional
        Transforms applied to ``input`` during training.
    eval_transform_x : callable, optional
        Transforms applied to ``input`` during validation and testing.
    transform_y : callable, optional
        Transforms applied to ``output``.
    shuffle : bool, default=False
        Only for ``train_dataloader``.
    train_batch_size : int, default=32
        ``batch_size`` for train dataloader.
    eval_batch_size : int, default=32
        ``batch_size`` for the validation and test dataloaders.
    config_dataloaders : dict, optional
        Dictionary for configuring the :class:`~torch.utils.data.DataLoader`'s.
        This is applied to all dataloaders, i.e.
        ``{train,validation,test}_dataloader``. For example::

            config_dataloaders = {
                'pin_memory': True,
                'num_workers': 2,
                }

    See Also
    --------
    :class:`~aidsorb.data.PCDDataset`
    :class:`~torch.utils.data.DataLoader` :
        For a description of ``shuffle``, ``batch_size`` and valid ``**kwargs``
        passed to ``config_dataloaders``.
    """
    def __init__(
            self, path_to_X: str, path_to_Y: str,
            index_col: str, labels: list,
            train_size: int=None,
            train_transform_x: Callable=None,
            eval_transform_x: Callable=None,
            transform_y: Callable=None,
            shuffle: bool=False,
            train_batch_size: int=32,
            eval_batch_size: int=32,
            config_dataloaders=None,
            ):
        super().__init__()
        
        self.path_to_X = path_to_X
        self.path_to_Y = path_to_Y

        self.index_col = index_col
        self.labels = labels

        self.train_transform_x = train_transform_x
        self.eval_transform_x = eval_transform_x
        self.transform_y = transform_y

        self.train_size = train_size
        self.shuffle = shuffle

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        if config_dataloaders is None:
            self.config_dataloaders = {}
        else:
            self.config_dataloaders = config_dataloaders

        self._path_to_names = Path(self.path_to_X).parent

    def setup(self, stage=None):
        r"""
        Setup train, validation and test datasets.
        """
        if stage in (None, 'fit'):
            # Load the names for training and validation.
            self._train_names = get_names(
                    os.path.join(self._path_to_names, 'train.json')
                    )[:self.train_size]  # Set the training set size.

            self._val_names = get_names(
                    os.path.join(self._path_to_names, 'validation.json')
                    )

            self.set_train_dataset()
            self.set_validation_dataset()

        if stage in (None, 'validate'):
            # Load the names for validation.
            self._val_names = get_names(
                    os.path.join(self._path_to_names, 'validation.json')
                    )

            self.set_validation_dataset()

        if stage in (None, 'test'):
            # Load the names for testing.
            self._test_names = get_names(
                    os.path.join(self._path_to_names, 'test.json')
                    )

            self.set_test_dataset()

    @property
    def train_names(self):
        r"""The names of point clouds used for training."""
        return self._train_names

    @property
    def val_names(self):
        r"""The names of point clouds used for validation."""
        return self._val_names

    @property
    def test_names(self):
        r"""The names of point clouds used for testing."""
        return self._test_names

    def set_train_dataset(self):
        r"""Setup the train dataset."""
        self.train_dataset = PCDDataset(
                pcd_names=self.train_names,
                path_to_X=self.path_to_X,
                path_to_Y=self.path_to_Y,
                index_col=self.index_col,
                labels=self.labels,
                transform_x=self.train_transform_x,
                transform_y=self.transform_y,
                )

    def set_validation_dataset(self):
        r"""Setup the validation dataset."""
        self.validation_dataset = PCDDataset(
                pcd_names=self.val_names,
                path_to_X=self.path_to_X,
                path_to_Y=self.path_to_Y,
                index_col=self.index_col,
                labels=self.labels,
                transform_x=self.eval_transform_x,
                transform_y=self.transform_y,
                )

    def set_test_dataset(self):
        r"""Setup the validation dataset."""
        self.test_dataset = PCDDataset(
                pcd_names=self.test_names,
                path_to_X=self.path_to_X,
                path_to_Y=self.path_to_Y,
                index_col=self.index_col,
                labels=self.labels,
                transform_x=self.eval_transform_x,
                transform_y=self.transform_y,
                )

    def train_dataloader(self):
        r"""
        Return the train dataloader.

        Can be called only after :meth:`setup` has been called and
        ``stage={None|fit}``.
        """
        return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=self.shuffle,
                **self.config_dataloaders,
                )

    def val_dataloader(self):
        r"""
        Return the validation dataloader.

        Can be called only after :meth:`setup` has been called and
        ``stage={None|fit|validate}``.
        """
        return DataLoader(
                dataset=self.validation_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                **self.config_dataloaders,
                )

    def test_dataloader(self):
        r"""
        Return the test dataloader.

        Can be called only after :meth:`setup` has been called and
        ``stage={None|test}``.
        """
        return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                **self.config_dataloaders,
                )
