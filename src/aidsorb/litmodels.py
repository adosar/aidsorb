r"""
Write the docstring for the module.

.. warning::
    Use 1 metric in test step.
"""
from typing import Callable
import torch
from torch.nn.functional import mse_loss
from torchmetrics.functional import r2_score, mean_absolute_error, mean_squared_error
import lightning as L
from aidsorb.models import PointNet


class PointNetLit(L.LightningModule):
    r"""
    Lightning module for :class:`aidsorb.models.PointNet`.

    Parameters
    ----------
    model : :class:`aidsorb.models.PointNet`
    loss : callable
        The loss function to be optimized during training.
        See `loss functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_.
    metric : callable
        The performance metric to be monitored.
        See `metrics <https://lightning.ai/docs/torchmetrics/stable/all-metrics.html>`_.
    lr : float, default=0.001
        The learning rate for :class:`torch.optim.Adam` optimizer.
    
    Examples
    --------
    >>> from aidsorb.models import PointNetClsHead, PointNet
    >>> from torch.nn.functional import mse_loss  # For regression.
    >>> from torchmetrics.functional import r2_score  # For regression.
    >>> model = PointNet(head=PointNetClsHead(n_outputs=10))
    >>> pointnetlit = PointNetLit(model=model, loss=mse_loss, metric=r2_score)
    >>> x = torch.randn(32, 5, 100)
    >>> y_pred, _ = pointnetlit(x)  # Ignore critical indices.
    >>> y_pred.shape
    torch.Size([32, 10])
    """
    def __init__(
            self, model: PointNet,
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
        return self.model(x)

    def training_step(self, batch, batch_idx):
        assert self.training
        assert torch.is_grad_enabled()

        x, y = batch
        y_pred, _ = self(x)
        loss = self.loss(y_pred, y)

        # Account for BatchNorm and Dropout.
        self.eval()
        with torch.no_grad():
            preds, _ = self(x)
        self.train()

        # Store for epoch-level operations.
        self.train_step_preds.append(preds.detach())
        self.train_step_targets.append(y)

        return loss

    def on_train_epoch_end(self):
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
        assert not self.training
        assert not torch.is_grad_enabled()

        x, y = batch
        preds, _ = self(x)

        # Store for epoch-level operations.
        self.val_step_preds.append(preds)
        self.val_step_targets.append(y)

    def on_validation_epoch_end(self):
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
        assert not self.training
        assert not torch.is_grad_enabled()

        x, y = batch
        preds, _ = self(x)

        # Store for epoch-level operations.
        self.test_step_preds.append(preds)
        self.test_step_targets.append(y)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_step_preds)
        targets = torch.cat(self.test_step_targets)

        metrics = {
                'r2_score': r2_score(preds=preds, target=targets),
                'mae': mean_absolute_error(preds=preds, target=targets),
                'mse': mean_squared_error(preds=preds, target=targets),
                }

        # Uncomment the following when API stabilizes.
        #metrics = {
        #        'Metric': self.metric(preds=preds, target=targets),
        #        'Loss': self.loss(preds=preds, target=targets),
        #        }

        self.log_dict(metrics)

        self.test_step_preds.clear()
        self.test_step_targets.clear()

    def predict_step(self, batch, batch_idx):
        assert not self.training
        assert not torch.is_grad_enabled()

        if len(batch) == 2:  # Batch with labels.
            x, y = batch
        else:
            x = batch  # Batch without labels.

        y_pred, _ = self(x)

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
