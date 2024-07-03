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
This model provides :class:`~lightning.pytorch.core.LightningModule`'s that can be
used with :bdg-link-primary:`PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`.

.. todo:
    Add support for :class:`torchmetrics.MetricCollection`.
"""
from typing import Callable
import torch
import lightning as L
from aidsorb.models import PointNet


class PointLit(L.LightningModule):
    r"""
    ``LightningModule`` for :class:`aidsorb.models`.

    Parameters
    ----------
    model : :class:`torch.nn.Module`
        Currently, only :class:`~aidsorb.models.PointNet` is supported.
    loss : callable
        The loss function to be optimized during training. For valid options,
        see `loss functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_.
    metric : callable
        The performance metric to be monitored. For valid options,
        see `metrics <https://lightning.ai/docs/torchmetrics/stable/all-metrics.html>`_.
    lr : float, default=0.001
        The learning rate for :class:`~torch.optim.Adam` optimizer.

    Examples
    --------
    >>> from aidsorb.modules import PointNetClsHead
    >>> from aidsorb.models import PointNet
    >>> from torch.nn.functional import mse_loss  # For regression.
    >>> from torchmetrics.functional import r2_score  # For regression.
    >>> model = PointNet(head=PointNetClsHead(n_outputs=10))
    >>> pointnetlit = PointLit(model=model, loss=mse_loss, metric=r2_score)
    >>> x = torch.randn(32, 5, 100)
    >>> out = pointnetlit(x)  # Ignore critical indices.
    >>> out.shape
    torch.Size([32, 10])
    """
    def __init__(
            self, model: torch.nn.Module,
            loss: Callable, metric: Callable,
            lr: float=1e-3
            ):
        super().__init__()

        self.model = model
        self.lr = lr
        
        self.loss = loss
        self.metric = metric

        # For epoch-level operations.
        self.train_step_preds = []
        self.train_step_targets = []

        self.val_step_preds = []
        self.val_step_targets = []

        self.test_step_preds = []
        self.test_step_targets = []

        self.save_hyperparameters(ignore=['model', 'loss', 'metric'])

    def forward(self, x):
        r"""
        Run forward pass (forward method) of ``model``.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        r"""
        Compute and return training loss on a single ``batch`` from the train set.

        Also make and store predictions on this single ``batch``.

        .. note::
            The ``BatchNorm`` and ``Dropout`` are set in inference mode during
            predictions, so an accurate estimate of training performance is
            reported.
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

        # Store for epoch-level operations.
        self.train_step_preds.append(preds.detach())
        self.train_step_targets.append(y)

        return loss

    def on_train_epoch_end(self):
        r"""
        Log ``metric`` calculated  on the whole train set.
        """
        preds = torch.cat(self.train_step_preds)
        targets = torch.cat(self.train_step_targets)

        metric = self.metric(preds=preds, target=targets)

        self.log('train_metric', metric, prog_bar=True)
        self.logger.experiment.add_scalars(
                'learning_curve', {'train_acc': metric},
                global_step=self.global_step,
                )

        self.train_step_preds.clear()
        self.train_step_targets.clear()

    def validation_step(self, batch, batch_idx):
        r"""
        Make and store predictions on a single ``batch`` from the validation set.
        """
        assert not self.training
        assert not torch.is_grad_enabled()

        x, y = batch
        preds = self(x)

        # Store for epoch-level operations.
        self.val_step_preds.append(preds)
        self.val_step_targets.append(y)

    def on_validation_epoch_end(self):
        r"""
        Log ``metric`` calculated  on the whole validation set.
        """
        preds = torch.cat(self.val_step_preds)
        targets = torch.cat(self.val_step_targets)

        metric = self.metric(preds=preds, target=targets)

        self.log('hp_metric', metric)
        self.log('val_metric', metric, prog_bar=True)
        self.logger.experiment.add_scalars(
                'learning_curve', {'val_acc': metric},
                global_step=self.global_step,
                )

        self.val_step_preds.clear()
        self.val_step_targets.clear()
        
    def test_step(self, batch, batch_idx):
        r"""
        Make and store predictions on a single ``batch`` from the test set.
        """
        assert not self.training
        assert not torch.is_grad_enabled()

        x, y = batch
        preds = self(x)

        # Store for epoch-level operations.
        self.test_step_preds.append(preds)
        self.test_step_targets.append(y)

    def on_test_epoch_end(self):
        r"""
        Return a :class:`dict` of ``loss`` and ``metric`` calculated
        on the whole test set.
        """
        preds = torch.cat(self.test_step_preds)
        targets = torch.cat(self.test_step_targets)

        # Add support for torchmetrics.Collection.
        metrics = {
                'Metric': self.metric(preds=preds, target=targets),
                'Loss': self.loss(input=preds, target=targets),
                }

        self.log_dict(metrics)

        self.test_step_preds.clear()
        self.test_step_targets.clear()

    def predict_step(self, batch, batch_idx):
        r"""
        Return predictions on a single ``batch``.
        """
        assert not self.training
        assert not torch.is_grad_enabled()

        if len(batch) == 2:  # Batch with labels.
            x = batch
        else:
            x = batch  # Batch without labels.

        y_pred = self(x)

        return y_pred

    def configure_optimizers(self):
        r""" Return the optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
