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
This module provides :class:`~lightning.pytorch.core.LightningModule`'s for use
with |lightning|.
"""
from typing import Callable
import torch
import torchmetrics
import lightning as L


class PointLit(L.LightningModule):
    r"""
    ``LightningModule`` for :class:`aidsorb.models`.

    .. _loss functions: https://pytorch.org/docs/stable/nn.html#loss-functions
    .. _monitor: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint

    Parameters
    ----------
    model : :class:`torch.nn.Module`
        Currently, only :class:`~aidsorb.models.PointNet` is available.
    loss : callable
        The loss function to be optimized during training. For valid options,
        see `loss functions`_.
    metric : :class:`torchmetrics.MetricCollection`
        The performance metric(s) to be logged and optionally monitored.

        .. note::
            The ``metric`` is logged on epoch-level.

        .. tip::
            You can use ``'val_<MetricName>'`` as the quantity to `monitor`_.
            For example, if ``metric=MetricCollection(R2Score(),
            MeanAbsoluteError())`` and you want to monitor
            :class:`~torchmetrics.R2Score`, configure the
            :class:`~lightning.pytorch.callbacks.ModelCheckpoint` as following::

                from lightning.pytorch.callbacks import ModelCheckpoint

                checkpoint_callback = ModelCheckpoint(monitor='val_R2Score', mode='max', ...)

    lr : float, default=0.001
        The learning rate for :class:`~torch.optim.Adam` optimizer.

    Examples
    --------
    >>> from aidsorb.modules import PointNetClsHead
    >>> from aidsorb.models import PointNet
    >>> from torch.nn import MSELoss
    >>> from torchmetrics import MetricCollection, R2Score, MeanAbsoluteError as MAE

    >>> model = PointNet(head=PointNetClsHead(n_outputs=10))
    >>> loss, metric = MSELoss(), MetricCollection(R2Score(), MAE())
    >>> litmodel = PointLit(model=model, loss=loss, metric=metric)

    >>> x = torch.randn(32, 5, 100)
    >>> out = litmodel(x)
    >>> out.shape
    torch.Size([32, 10])
    """
    def __init__(
            self, model: torch.nn.Module,
            loss: Callable, metric: torchmetrics.MetricCollection,
            lr: float=1e-3
            ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.loss = loss
        self.metric = metric

        # Ignore nn.Modules for reducing the size of checkpoints.
        self.save_hyperparameters(ignore=['model', 'loss', 'metric'])

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
        Compute and return training loss on a single ``batch`` from the train set.

        Also, make predictions that will be used on epoch-level operations.

        .. note::
            Inference mode is enabled during predictions, so an accurate
            estimate of training performance (e.g. when using
            :class:`~torch.nn.Dropout`) is reported.
        """
        assert self.training
        assert torch.is_grad_enabled()

        x, y = batch
        y_pred = self(x)
        loss = self.loss(input=y_pred, target=y)

        # Account for BatchNorm and Dropout.
        self.eval()
        with torch.no_grad():
            preds = self(x)
        self.train()

        # Log metric on epoch-level.
        self.train_metric.update(preds=preds, target=y)
        self.log_dict(self.train_metric, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        r"""
        Make predictions on a single ``batch`` from the validation set for epoch-level
        operations.
        """
        assert not self.training
        assert not torch.is_grad_enabled()

        x, y = batch
        preds = self(x)

        # Log metric on epoch-level.
        self.val_metric.update(preds=preds, target=y)
        self.log_dict(self.val_metric, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        r"""
        Make predictions on a single ``batch`` from the test set for epoch-level
        operations.
        """
        assert not self.training
        assert not torch.is_grad_enabled()

        x, y = batch
        preds = self(x)

        # Log metric on epoch-level.
        self.test_metric.update(preds=preds, target=y)
        self.log_dict(self.test_metric, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        r"""
        Return predictions on a single ``batch``.
        """
        assert not self.training
        assert not torch.is_grad_enabled()

        if len(batch) == 2:  # Batch with labels.
            x, _ = batch
        else:
            x = batch  # Batch without labels.

        return self(x)

    def configure_optimizers(self):
        r""" Return the optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
