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
:class:`~lightning.pytorch.core.LightningDataModule`'s for use with |lightning|.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
from . data import get_names, PCDDataset


class PCDDataModule(L.LightningDataModule):
    r"""
    ``LightningDataModule`` for the :class:`~.PCDDataset`.

    This datamodule assumes the following directory structure::

        project_root
        ├── source      <-- path_to_X
        │   ├── foo.npy
        │   ├── ...
        │   └── bar.npy
        ├── test.json
        ├── train.json
        └── validation.json

    and setups the train, validation, and test datasets, all of which are
    instances of :class:`~.PCDDataset`.

    .. note::
        For validation and test dataloaders, ``shuffle=False`` and
        ``drop_last=False``.

    .. warning::
        * Comma ``,`` is assumed as the field separator in ``.csv`` file.
        * If ``train_size`` is specified, the first ``train_size`` point clouds
          from ``train.json`` will be used. **If the data were not split with**
          :func:`~aidsorb.data.prepare_data`, **ensure that names in**
          ``train.json`` **don't follow a particular order**.

    .. todo::
        Add support for ``predict_dataloader``.

    Parameters
    ----------
    path_to_X : str
        Absolute or relative path to the directory holding the point clouds.
    path_to_Y : str, optional
        Absolute or relative path to the ``.csv`` file holding the labels of the
        point clouds.
    index_col : str, optional
        Column name of the ``.csv`` file to be used for indexing.
    labels : sequence, optional
        Column names of the ``.csv`` file containing the properties to be
        predicted.
    train_size : int, optional
        Number of training samples. If ``None``, all training samples are used.
    train_transform_x : callable, optional
        Transformation to apply to point cloud during training.
    eval_transform_x : callable, optional
        Transformation to apply to point cloud during validation and testing.
    transform_y : callable, optional
        Transformation to apply to label.
    shuffle : bool, default=False
        Only for train dataloader.
    drop_last : bool, default=False
        Only for train dataloader.
    train_batch_size : int, default=32
        Batch size for train dataloader.
    eval_batch_size : int, default=32
        Batch size for validation and test dataloaders.
    config_dataloaders : dict, optional
        Dictionary for configuring all dataloaders. For example::

            config_dataloaders = {
                'pin_memory': True,
                'num_workers': 2,
                }

    See Also
    --------
    :class:`~torch.utils.data.DataLoader` :
        For a description of ``shuffle``, ``drop_last`` and valid options for
        ``config_dataloaders``.
    """
    def __init__(
            self,
            path_to_X: str,
            path_to_Y: str = None,
            index_col: str = None,
            labels: Sequence = None,
            train_size: int = None,
            train_transform_x: Callable = None,
            eval_transform_x: Callable = None,
            transform_y: Callable = None,
            shuffle: bool = False,
            drop_last: bool = False,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
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
        self.drop_last = drop_last

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        # Options passed to all dataloaders.
        self.config_dataloaders = {}
        if config_dataloaders is not None:
            self.config_dataloaders = config_dataloaders

        # For convenience with load_from_checkpoint.
        self.save_hyperparameters()

        self._path_to_names = Path(self.path_to_X).parent

    def setup(self, stage=None):
        r"""
        Setup train, validation and test datasets.

        .. tip::
            Datasets are accesible via ``self.{train,validation,test}_dataset``.

        Parameters
        ----------
        stage : {None, 'fit', 'validate', 'test'}, optional
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
        return tuple(self._train_names)

    @property
    def val_names(self):
        r"""The names of point clouds used for validation."""
        return tuple(self._val_names)

    @property
    def test_names(self):
        r"""The names of point clouds used for testing."""
        return tuple(self._test_names)

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
        r"""Setup the test dataset."""
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
        ``stage`` is ``{None, 'fit'}``.

        Returns
        -------
            :class:`~torch.utils.data.DataLoader`
        """
        return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                **self.config_dataloaders,
                )

    def val_dataloader(self):
        r"""
        Return the validation dataloader.

        Can be called only after :meth:`setup` has been called and
        ``stage`` is ``{None, 'fit', 'validate'}``.

        Returns
        -------
            :class:`~torch.utils.data.DataLoader`
        """
        return DataLoader(
                dataset=self.validation_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=False,
                **self.config_dataloaders,
                )

    def test_dataloader(self):
        r"""
        Return the test dataloader.

        Can be called only after :meth:`setup` has been called and
        ``stage`` is ``{None, 'test'}``.

        Returns
        -------
            :class:`~torch.utils.data.DataLoader`
        """
        return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=False,
                **self.config_dataloaders,
                )
