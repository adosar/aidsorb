r"""
Add module docstring.
"""

import os
from pathlib import Path
import lightning as L
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
        See :class:`PCDDataset`.
    path_to_Y : str
        See :class:`PCDDataset`.
    index_col : str
        See :class:`PCDDataset`.
    labels : list
        See :class:`PCDDataset`.
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
    train_batch_size : int, default=64
        Only for ``train_dataloader``. See `DataLoader`_.
    eval_batch_size : int, default=64
        For ``{validation,test,predict}_dataloader``. See `DataLoader`_.
    **kwargs
        Valid keyword arguments for `DataLoader`_. For ``*_dataloader``. See
        `DataLoader`_.

    .. _DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    def __init__(
            self, path_to_X, path_to_Y, index_col,
            labels, train_size=None,
            train_transform_x=None, eval_transform_x=None,
            transform_y=None, shuffle=False,
            train_batch_size=32, eval_batch_size=32, **kwargs
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
        self._config_dataloader = kwargs

        self._path_to_names = Path(self.path_to_X).parent

    def setup(self, stage=None):
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

        if stage in (None, 'test'):
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

    def set_train_dataset(self):
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
        return DataLoader(
                dataset=self.train_dataset,
                shuffle=self.shuffle,
                batch_size=self.train_batch_size,
                **self._config_dataloader,
                )

    def val_dataloader(self):
        return DataLoader(
                dataset=self.validation_dataset,
                batch_size=self.eval_batch_size,
                **self._config_dataloader,
                )

    def test_dataloader(self):
        return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.eval_batch_size,
                **self._config_dataloader,
                )
