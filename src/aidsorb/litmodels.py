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
from torch.nn import Module
from torchmetrics import MetricCollection
import lightning as L
from . _torch_utils import get_optimizers


class PCDLit(L.LightningModule):
    r"""
    :class:`~lightning.pytorch.core.LightningModule` for supervised learning on
    point clouds.

    This module supports regression and classification tasks.

    .. note::
        * Parametes for which ``requires_grad=False`` are excluded from
          optimization.
        * ``{train,validation,test}_step`` methods expect a batch of the form
          ``(x, y)``, where ``x`` is compatible with :meth:`PCDLit.forward` and
          ``y`` is compatible with output of :meth:`PCDLit.forward` as required by
          ``criterion`` and ``metric``. For :meth:`PCDLit.predict_step`, ``y`` is
          ignored.

    .. tip::
        You can use ``'val_<MetricName>'`` as the quantity to monitor. For
        example, if::

            from torchmetrics import R2Score, MeanAbsoluteError, MetricCollection
            metric = MetricCollection(R2Score(), MeanAbsoluteError())

        and you want to monitor ``R2Score``, configure the
        :class:`~lightning.pytorch.callbacks.ModelCheckpoint` as following::

            from lightning.pytorch.callbacks import ModelCheckpoint
            checkpoint_callback = ModelCheckpoint(monitor='val_R2Score', mode='max', ...)


    .. _logging: https://lightning.ai/docs/pytorch/stable/extensions/logging.html#automatic-logging

    Parameters
    ----------
    model : torch.nn.Module
        Architecture for point cloud processing.
    criterion : callable
        Loss function to be optimized during training.
    metric : torchmetrics.MetricCollection
        Metric(s) to be logged and optionally monitored. All metric(s) are
        logged based on Lightning's default hooks. For more information, see
        `logging`_.
    config_optimizer : dict, default=None
        Dictionary for configuring optimizer. If :obj:`None`, the
        :class:`~torch.optim.Adam` optimizer with default hyperparameters is
        used.
        
        * ``'name'`` optimizer's class name :class:`str`
        * ``'hparams'`` optimizer's hyperparameters :class:`dict`

    config_scheduler : dict, optional
        Dictionary for configuring learning rate scheduler.
        
        * ``'name'`` scheduler's class name :class:`str`
        * ``'hparams'`` scheduler's hyperparameters :class:`dict`
        * ``'config'`` scheduler's config  :class:`dict`

    Examples
    --------
    >>> import torch
    >>> from aidsorb.modules import PointNetClsHead, PointNet
    >>> from torchmetrics import MetricCollection, R2Score, MeanAbsoluteError as MAE

    >>> model = PointNet(head=PointNetClsHead(n_outputs=10))
    >>> criterion, metric = torch.nn.MSELoss(), MetricCollection(R2Score(), MAE())

    >>> # Adam optimizer with default hyperparameters, no scheduler.
    >>> litmodel = PCDLit(model, criterion, metric)

    >>> # Custom optimizer and scheduler.
    >>> config_optimizer = {
    ... 'name': 'SGD',
    ... 'hparams': {'lr': 0.1},
    ... }
    >>> config_scheduler = {
    ... 'name': 'StepLR',
    ... 'hparams': {'step_size': 2},
    ... 'config': {'interval': 'step'},
    ... }
    >>> litmodel = PCDLit(model, criterion, metric, config_optimizer, config_scheduler)

    >>> # Forward pass.
    >>> x = torch.randn(32, 5, 100)
    >>> litmodel(x).shape
    torch.Size([32, 10])
    """
    def __init__(
            self,
            model: Module,
            criterion: Callable,
            metric: MetricCollection,
            config_optimizer: dict=None,
            config_scheduler: dict=None,
            ):
        super().__init__()
        self.save_hyperparameters()  # For argument-less load_from_checkpoint.

        self.model = model
        self.criterion = criterion
        self.metric = metric

        self.config_optimizer = config_optimizer
        self.config_scheduler = config_scheduler

        # For logging metric(s) at different stages.
        self.train_metric = metric.clone(prefix='train_')
        self.val_metric = metric.clone(prefix='val_')
        self.test_metric = metric.clone(prefix='test_')

    def forward(self, x):
        r"""Run forward pass (forward method) of ``model``."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        r"""
        Return loss on a single batch from the train set and log metric(s).

        .. note::
            Training metric(s) is computed with training mode enabled and thus,
            may underestimate the true training performance if ``model``
            contains modules like ``Dropout`` etc.
        """
        x, y = batch
        preds = self(x)

        # Compute training loss.
        loss = self.criterion(input=preds, target=y)

        # Log metric on step-level.
        self.train_metric(preds=preds, target=y)
        self.log_dict(self.train_metric, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        r"""
        Make predictions on a single batch from the validation set and log
        metric(s).
        """
        x, y = batch
        preds = self(x)

        # Log metric on epoch-level.
        self.val_metric.update(preds=preds, target=y)
        self.log_dict(self.val_metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        r"""
        Make predictions on a single batch from the test set and log metric(s).
        """
        x, y = batch
        preds = self(x)

        # Log metric on epoch-level.
        self.test_metric.update(preds=preds, target=y)
        self.log_dict(self.test_metric, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        r"""Return predictions on a single batch."""
        x, _ = batch

        return self(x)

    def configure_optimizers(self):
        r"""
        Configure optimizer and optionally learning rate scheduler.

        Returns
        -------
        optimizers : :class:`~torch.optim.Optimizer` or dict
            Single optimizer if ``scheduler=None``, else dictionary with keys:
            ``'optimizer'`` and ``'lr_scheduler'``.

        Examples
        --------
        >>> import torch
        >>> from torchmetrics import MetricCollection, R2Score
        >>> criterion = torch.nn.MSELoss()
        >>> metric = MetricCollection(R2Score())

        >>> model = torch.nn.Linear(2, 2)
        >>> litmodel = PCDLit(model, criterion, metric)
        >>> litmodel.configure_optimizers()
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        Adam (
            ...
        )

        >>> model = torch.nn.Linear(4, 4)
        >>> _ = model.requires_grad_(False)
        >>> litmodel = PCDLit(model, criterion, metric)
        >>> litmodel.configure_optimizers()
        Traceback (most recent call last):
            ...
        ValueError: optimizer got an empty parameter list
        """
        # Extract trainable parameters (i.e. requires_grad=True).
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        return get_optimizers(
                params=params,
                config_optim=self.config_optimizer,
                config_lrs=self.config_scheduler,
                )
