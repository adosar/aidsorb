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
:class:`~lightning.pytorch.core.LightningModule`'s for use with |lightning|.
"""
from collections.abc import Callable
import torch
import torchmetrics
import lightning as L


class PCDLit(L.LightningModule):
    r"""
    ``LightningModule`` for supervised learning on point clouds.

    .. note::
        * ``metric`` is logged on epoch-level.
        * ``*_step`` methods expect a ``batch`` of the form ``(pcds, labels)``.

    .. tip::
        You can use ``'val_<MetricName>'`` as the quantity to `monitor`_.
        For example, if ``metric=MetricCollection(R2Score(),
        MeanAbsoluteError())`` and you want to monitor ``R2Score``, configure
        the :class:`~lightning.pytorch.callbacks.ModelCheckpoint` as following::

            from lightning.pytorch.callbacks import ModelCheckpoint

            checkpoint_callback = ModelCheckpoint(monitor='val_R2Score', mode='max', ...)

    .. _monitor: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint

    Parameters
    ----------
    model : :class:`~torch.nn.Module`
        Custom architecture or one from :mod:`aidsorb.models`.
    criterion : callable
        Loss function to be optimized during training.
    metric : :class:`~torchmetrics.MetricCollection`
        Metric(s) to be logged and optionally monitored.
    lr : float, default=0.001
        Learning rate for :class:`~torch.optim.Adam` optimizer.

    Examples
    --------
    >>> from aidsorb.modules import PointNetClsHead
    >>> from aidsorb.models import PointNet
    >>> from torch.nn import MSELoss
    >>> from torchmetrics import MetricCollection, R2Score, MeanAbsoluteError as MAE

    >>> model = PointNet(head=PointNetClsHead(n_outputs=10))
    >>> criterion, metric = MSELoss(), MetricCollection(R2Score(), MAE())
    >>> litmodel = PCDLit(model=model, criterion=criterion, metric=metric)

    >>> x = torch.randn(32, 5, 100)
    >>> out = litmodel(x)
    >>> out.shape
    torch.Size([32, 10])
    """
    def __init__(
            self, model: torch.nn.Module,
            criterion: Callable, metric: torchmetrics.MetricCollection,
            lr: float=1e-3
            ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.criterion = criterion
        self.metric = metric

        # Ignore nn.Modules for reducing the size of checkpoints.
        self.save_hyperparameters(ignore=['model', 'criterion', 'metric'])

        # For epoch-level operations.
        self.train_metric = metric.clone(prefix='train_')
        self.val_metric = metric.clone(prefix='val_')
        self.test_metric = metric.clone(prefix='test_')

    def forward(self, x):
        r"""
        Run forward pass (forward method) of ``model``.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        r"""
        Compute and return training loss on a single batch from the train set.

        Also, make predictions that will be used on epoch-level operations.

        .. note::
            Training loss is computed with training mode enabled and thus, may
            underestimate the true training loss if ``model`` contains
            modules like ``Dropout`` etc.
        """
        x, y = batch
        preds = self(x)

        # Compute training loss.
        loss = self.criterion(input=preds, target=y)

        # Log metric on epoch-level.
        self.train_metric.update(preds=preds, target=y)
        self.log_dict(self.train_metric, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        r"""
        Make predictions on a single batch from the validation set for epoch-level
        operations.
        """
        x, y = batch
        preds = self(x)

        # Log metric on epoch-level.
        self.val_metric.update(preds=preds, target=y)
        self.log_dict(self.val_metric, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        r"""
        Make predictions on a single batch from the test set for epoch-level
        operations.
        """
        x, y = batch
        preds = self(x)

        # Log metric on epoch-level.
        self.test_metric.update(preds=preds, target=y)
        self.log_dict(self.test_metric, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        r"""
        Return predictions on a single batch.
        """
        x, _ = batch

        return self(x)

    def configure_optimizers(self):
        r""" Return the optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
