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
from typing import Any
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from .data import PCDDataset, get_names


class PCDDataModule(L.LightningDataModule):
    r"""
    LightningDataModule for supervised/unsupervised learning on point clouds.

    Given the following directory structure::

        project_root
        ├── source      <-- path_to_X
        │   ├── foo.npy
        │   ├── ...
        │   └── bar.npy
        ├── test.json
        ├── train.json
        └── validation.json

    train, validation, and test datasets are set up, all of which are instances
    of :class:`~.PCDDataset`.

    .. note::
        Comma ``,`` is assumed as the field separator in ``.csv`` file.

    .. warning::
        * For validation and test dataloaders, ``shuffle=False`` and
          ``drop_last=False``.
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
    labels : list, optional
        Column names of the ``.csv`` file containing the properties to be
        predicted.
    train_size : int, default=None
        Number of training samples. If :obj:`None`, all training samples are used.
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

        .. note::
            The dictionary is not copied. To avoid side effects, consider
            passing a copy.

    See Also
    --------
    :class:`~torch.utils.data.DataLoader` :
        For a description of ``shuffle``, ``drop_last`` and valid options for
        ``config_dataloaders``.
    """
    def __init__(
            self,
            path_to_X: str,
            *,
            path_to_Y: str | None = None,
            index_col: str | None = None,
            labels: list[str] | None = None,
            train_size: int | None = None,
            train_transform_x: Callable | None = None,
            eval_transform_x: Callable | None = None,
            transform_y: Callable | None = None,
            shuffle: bool = False,
            drop_last: bool = False,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            config_dataloaders: dict[str, Any] | None = None,
            ) -> None:
        super().__init__()
        self.save_hyperparameters()  # For argument-less load_from_checkpoint.
        
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

        # Configuration for all dataloaders.
        self.config_dataloaders = {}
        if config_dataloaders is not None:
            self.config_dataloaders = config_dataloaders

    def setup(self, stage: str | None = None) -> None:
        r"""
        Set up train, validation and test datasets.

        .. tip::
            Datasets are accesible via ``self.{train,validation,test}_dataset``.

        Parameters
        ----------
        stage : {None, 'fit', 'validate', 'test'}, default=None
            Which datasets to set up.

            * If ``'fit'``, only the train and validation datasets are set up.
            * If ``'validate'`` or ``'test'``, only the corresponding dataset is set up.
            * If :obj:`None`, all datasets are set up.
        """
        if stage == 'fit':
            self._setup_dataset('train')
            self._setup_dataset('validation')
        if stage == 'validate':
            self._setup_dataset('validation')
        if stage == 'test':
            self._setup_dataset('test')
        if stage is None:
            for mode in ['train', 'validation', 'test']:
                self._setup_dataset(mode)

    def _setup_dataset(self, mode: str) -> None:
        path_to_names = Path(self.path_to_X).parent
        pcd_names = get_names(os.path.join(path_to_names, f'{mode}.json'))

        if mode == 'train':
            transform_x = self.train_transform_x
            pcd_names = pcd_names[:self.train_size]  # Set the training set size.
        else:
            transform_x = self.eval_transform_x

        dataset = PCDDataset(
                pcd_names=pcd_names,
                path_to_X=self.path_to_X,
                path_to_Y=self.path_to_Y,
                index_col=self.index_col,
                labels=self.labels,
                transform_x=transform_x,
                transform_y=self.transform_y,
                )
        setattr(self, f'{mode}_dataset', dataset)

    def train_dataloader(self) -> DataLoader:
        r"""
        Return the train dataloader.

        Can be called only after :meth:`setup` has been called and
        ``stage`` is ``{None, 'fit'}``.

        Returns
        -------
            :class:`~torch.utils.data.DataLoader`
        """
        # pylint: disable=no-member
        return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                **self.config_dataloaders,
                )

    def val_dataloader(self) -> DataLoader:
        r"""
        Return the validation dataloader.

        Can be called only after :meth:`setup` has been called and
        ``stage`` is ``{None, 'fit', 'validate'}``.

        Returns
        -------
            :class:`~torch.utils.data.DataLoader`
        """
        # pylint: disable=no-member
        return DataLoader(
                dataset=self.validation_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=False,
                **self.config_dataloaders,
                )

    def test_dataloader(self) -> DataLoader:
        r"""
        Return the test dataloader.

        Can be called only after :meth:`setup` has been called and
        ``stage`` is ``{None, 'test'}``.

        Returns
        -------
            :class:`~torch.utils.data.DataLoader`
        """
        # pylint: disable=no-member
        return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                drop_last=False,
                **self.config_dataloaders,
                )
