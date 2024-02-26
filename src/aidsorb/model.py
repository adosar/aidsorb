import torch
import lightning as L
from torch import nn, optim


class STNkd(nn.Module):
    def __init__(self, dim=64):  # The embedding dimension.
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv1d(dim, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU()
                )
        self.conv2 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU()
                )
        self.conv3 = nn.Sequential(
                nn.Conv1d(128, 1024, kernel_size=1),
                nn.BatchNorm1d(1024),
                nn.ReLU()
                )
        self.fc1 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                )
        self.fc3 = nn.Sequential(
                nn.Linear(256, dim*dim),
                )

        self.dim = dim

    def forward(self, x):
        # The input should be batched.
        bs = x.shape[0]  # The batch size.

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2)[0]  # Keep only the values.

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # Initialize the identity matrix.
        identity = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)

        x = x.view(-1, self.dim, self.dim) + identity

        return x


class DummyModel(L.LightningModule):
    def __init__(self, n_features, n_outputs, loss, metric):
        super().__init__()

        # Here is the architecture.
        self.fc1 = torch.nn.Linear(n_features, n_outputs)

        # The loss and metric to be tracked.
        self.loss = loss
        self.metric = metric

    def forward(self, x):
        out = self.fc1(x)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(input=y_hat, target=y)
        metric = self.metric(preds=y_hat, target=y)

        values = {'train_loss': loss, 'train_metric': metric}
        self.log_dict(values, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(input=y_hat, target=y)
        metric = self.metric(preds=y_hat, target=y)

        values = {'val_loss': loss, 'val_metric': metric}
        self.log_dict(values, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3)
