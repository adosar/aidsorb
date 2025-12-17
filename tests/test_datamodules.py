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
Unit tests for the aidsorb.datamodules module.

Run from: project's root directory
Command: python -m unittest tests.test_datamodules
"""

import unittest
from itertools import combinations

from torch.utils.data import RandomSampler, SequentialSampler

from aidsorb.data import Collator, get_names
from aidsorb.datamodules import PCDDataModule
from aidsorb.transforms import Center, RandomRotation


def dummy_tfm(x):
    return x - 1


class TestPCDDataModule(unittest.TestCase):
    def setUp(self):
        # Arguments for the datamodule.
        self.train_size = 2
        self.train_trans_x = Center()
        self.eval_trans_x = RandomRotation()
        self.trans_y = dummy_tfm
        self.shuffle = True
        self.drop_last = True
        self.train_bs = 3
        self.eval_bs = 2
        self.config_dataloaders = {
                'pin_memory': True,
                'num_workers': 0,
                'collate_fn': Collator(channels_first=True)
                }

        # Instantiate the datamodule.
        self.dm = PCDDataModule(
                path_to_X='tests/dummy/toy_project/pcd_data',
                path_to_Y='tests/dummy/toy_dataset.csv',
                index_col='id',
                labels=['y2', 'y3'],
                train_size=self.train_size,
                train_transform_x=self.train_trans_x,
                eval_transform_x=self.eval_trans_x,
                transform_y=self.trans_y,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                train_batch_size=self.train_bs,
                eval_batch_size=self.eval_bs,
                config_dataloaders=self.config_dataloaders,
                )

    def test_setup_all(self):
        self.dm.setup()
        for mode in ['train', 'validation', 'test']:
            self.assertTrue(hasattr(self.dm, f'{mode}_dataset'))

    def test_setup_fit(self):
        self.dm.setup('fit')
        for mode in ['train', 'validation']:
            self.assertTrue(hasattr(self.dm, f'{mode}_dataset'))
        self.assertFalse(hasattr(self.dm, 'test_dataset'))

    def test_setup_val(self):
        self.dm.setup('validate')
        self.assertTrue(hasattr(self.dm, 'validation_dataset'))
        for mode in ['train', 'test']:
            self.assertFalse(hasattr(self.dm, f'{mode}_dataset'))

    def test_setup_test(self):
        self.dm.setup('test')
        self.assertTrue(hasattr(self.dm, 'test_dataset'))
        for mode in ['train', 'validation']:
            self.assertFalse(hasattr(self.dm, f'{mode}_dataset'))

    def test_datasets(self):
        self.dm.setup()

        # Check that the datasets have the correct size.
        val_names = get_names('tests/dummy/toy_project/validation.json')
        test_names = get_names('tests/dummy/toy_project/test.json')

        self.assertEqual(len(self.dm.train_dataset), self.train_size)
        self.assertEqual(len(self.dm.validation_dataset), len(val_names))
        self.assertEqual(len(self.dm.test_dataset), len(test_names))

        # The pairwise intersections must be the empty set.
        for ds_comb in combinations([
                self.dm.train_dataset.pcd_names,
                self.dm.validation_dataset.pcd_names,
                self.dm.test_dataset.pcd_names,
                ], r=2):
            self.assertEqual(set(ds_comb[0]) & set(ds_comb[1]), set())

        # Check that the transformations are passed correcly.
        for ds in (
                self.dm.train_dataset,
                self.dm.validation_dataset,
                self.dm.test_dataset,
                ):
            if ds is self.dm.train_dataset:
                self.assertIs(ds.transform_x, self.train_trans_x)
            else:
                self.assertIs(ds.transform_x, self.eval_trans_x)

            self.assertIs(ds.transform_y, self.trans_y)

    def test_dataloaders(self):
        self.dm.setup()

        dataloaders = [
                self.dm.train_dataloader(),
                self.dm.val_dataloader(),
                self.dm.test_dataloader(),
                ]
        datasets = [
                self.dm.train_dataset,
                self.dm.validation_dataset,
                self.dm.test_dataset,
                ]

        for i, dl in enumerate(dataloaders):
            # Check that dataloaders use appropriate settings.
            if i == 0:
                self.assertIsInstance(dl.sampler, RandomSampler)
                self.assertIs(dl.batch_size, self.train_bs)
                self.assertIs(dl.drop_last, self.drop_last)
            else:
                self.assertIsInstance(dl.sampler, SequentialSampler)
                self.assertIs(dl.batch_size, self.eval_bs)

            self.assertIs(dl.collate_fn, self.config_dataloaders['collate_fn'])

            # Check that collate function is used properly.
            for x, y in dl:
                self.assertEqual(x.ndim, 3)
                self.assertEqual(y.ndim, 2)

        # Check that each dataloaders loads the correct dataset.
        for dl, ds in zip(dataloaders, datasets):
            self.assertIs(dl.dataset, ds)
