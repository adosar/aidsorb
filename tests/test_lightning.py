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
Unit tests for testing AIdsorb with Lightning.

Run from: project's root directory
Command: python -m unittest tests.test_lightning
"""

import os
import tempfile
import unittest
import torch
import lightning as L
from torchmetrics import MetricCollection, MeanAbsoluteError as MAE
from aidsorb.data import Collator
from aidsorb.litmodels import PCDLit
from aidsorb.datamodules import PCDDataModule


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.LazyLinear(1)

    def forward(self, x):
        x = x[..., 0]  # Reduce to shape (N, C).
        return self.layer(x)


class TestLightning(unittest.TestCase):
    def setUp(self):
        # Temporary directory for lightning logs.
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')

        dummy_model = DummyModel()
        criterion, metric = torch.nn.MSELoss(), MetricCollection(MAE())

        config_optimizer = {
                'name': 'SGD',
                'hparams': {'lr': 1},
                }
        config_scheduler = {
                'name': 'ConstantLR',
                'hparams': dict(),  # Default hyperparameters.
                'config': {'interval': 'step'},
                }

        self.datamodule = PCDDataModule(
                path_to_X='tests/dummy/toy_project/pcd_data',
                path_to_Y='tests/dummy/toy_dataset.csv',
                index_col='id',
                labels=['y1'],
                config_dataloaders={'collate_fn': Collator()},
                )

        self.litmodel = PCDLit(
                model=dummy_model,
                criterion=criterion,
                metric=metric,
                config_optimizer=config_optimizer,
                config_scheduler=config_scheduler,
                )

        self.trainer = L.Trainer(max_epochs=1, default_root_dir=self.tempdir.name)

    def test_lightning(self):
        # Check training loop.
        self.trainer.fit(self.litmodel, self.datamodule)

        # Check that optimizers are configured correctly.
        self.assertIsInstance(
                self.litmodel.optimizers().optimizer,
                torch.optim.SGD
                )
        self.assertIsInstance(
                self.litmodel.lr_schedulers(),
                torch.optim.lr_scheduler.ConstantLR,
                )

        # Check validation and test loops.
        self.trainer.validate(self.litmodel, self.datamodule)
        self.trainer.test(self.litmodel, self.datamodule)

        # Get path to a checkpoint.
        ckpt_dir = f'{self.tempdir.name}/lightning_logs/version_0/checkpoints'
        ckpt_name = os.listdir(ckpt_dir)[0]
        ckpt_path = f'{ckpt_dir}/{ckpt_name}'

        # Check that modules can be loaded from checkpoint.
        for Module in [PCDDataModule, PCDLit]:
            Module.load_from_checkpoint(ckpt_path)

    def tearDown(self):
        self.tempdir.cleanup()
