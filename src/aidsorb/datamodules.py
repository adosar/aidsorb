r"""
Add module docstring.
"""
import os
from pathlib import Path
import lightning as L
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from . data import get_names, PCDDataset


class PCDDataModule(L.LightningDataModule):
    r"""
    DataModule for point clouds.

    The following directory structure is assumed::

        pcd_data
        ├──pcds.npz  <- path_to_X
        ├──train.json
        ├──validation.json
        └──test.json

    .. note::
        In order to use this module you must have already prepared your data
        with :func:`prepare_data`.

    Parameters
    ----------
    path_to_X : str
        Absolute or relative path to the ``.npz`` file holding the point clouds.
    path_to_Y : str
        Absolute or relative path to the ``.csv`` file holding the labels of
        the point clouds.
    index_col : str
        Column name of the ``.csv`` file to be used as row labels.
    labels : list
        List containing the names of the properties to be predicted.
    train_size : int, optional
        The number of training samples. By default, all training samples are
        used.
    train_transform_x : callable, optional
        Transforms applied to the point clouds. See :class:`PCDDataset`.

        .. note::
            These will be applied only during training.

    eval_transform_x : callable, optional
        Transforms applied to the point clouds. See :class:`PCDDataset`.

        .. note::
            These will be applied during validation, testing and prediction.

    transform_y : callable, optional
        Transforms applied to the labels. See :class:`PCDDataset`.
    shuffle : bool, default=False
        Only for ``train_dataloader``. See `DataLoader`_.
    batch_size : int, default=64
        See `DataLoader`_.
    kwargs : dict, optional
        Valid keyword arguments for `DataLoader`_.

    .. _DataLoader : https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    def __init__(
            self, path_to_X, path_to_Y, index_col,
            labels, train_size=None,
            train_transform_x=None, eval_transform_x=None,
            transform_y=None, shuffle=False,
            batch_size=64, **kwargs
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
        self.batch_size = batch_size

        self._config_dataloader = kwargs

    def prepare_data(self):
        self._path_to_names = Path(self.path_to_X).parent
        self._pcds_npz = np.load(self.path_to_X, mmap_mode='r')
        self._df = pd.read_csv(self.path_to_Y, index_col=self.index_col)

    def setup(self, stage):
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

        if stage in (None, 'test', 'predict'):
            # Load the names for testing and prediction.
            self._test_names = get_names(
                    os.path.join(self._path_to_names, 'test.json')
                    )

            self.set_test_dataset()

    @property
    def train_names(self):
        return self._train_names

    @property
    def val_names(self):
        return self._val_names

    @property
    def test_names(self):
        return self._test_names

    @property
    def predict_names(self):
        return self._test_names

    def set_train_dataset(self):
        self.train_dataset = PCDDataset(
                pcd_names=self.train_names,
                pcd_X=self._pcds_npz, pcd_Y=self._df,
                labels=self.labels,
                transform_x=self.train_transform_x,
                transform_y=self.transform_y,
                )

    def set_validation_dataset(self):
        self.validation_dataset = PCDDataset(
                pcd_names=self.val_names,
                pcd_X=self._pcds_npz, pcd_Y=self._df,
                labels=self.labels,
                transform_x=self.eval_transform_x,
                transform_y=self.transform_y,
                )

    def set_test_dataset(self):
        return PCDDataset(
                pcd_names=self.test_names,
                pcd_X=self._pcds_npz, pcd_Y=self._df,
                labels=self.labels,
                transform_x=self.eval_transform_x,
                transform_y=self.transform_y,
                )

    def train_dataloader(self):
        return DataLoader(
                dataset=self.train_dataset,
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                **self._config_dataloader,
                )

    def validation_dataloader(self):
        return DataLoader(
                dataset=self.validation_dataset,
                batch_size=self.batch_size,
                **self._config_dataloader,
                )

    def test_dataloader(self):
        return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size,
                **self._config_dataloader,
                )

    def predict_dataloader(self):
        return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size,
                **self._config_dataloader,
                )
