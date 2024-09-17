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
Unit tests for the aidsorb.data module.

Run from: project's root directory
Command: python -m unittest tests.test_data
"""

import os
import doctest
import unittest
import tempfile
from itertools import combinations
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from aidsorb import data
from aidsorb.utils import pcd_from_dir
from aidsorb.transforms import Center
from aidsorb.data import (
        get_names, prepare_data, PCDDataset,
        pad_pcds, Collator
        )


class TestPrepareData(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        self.outname = os.path.join(self.tempdir.name, 'pcds.npz')
        self.split_ratio = (2, 1, 3)

        pcd_from_dir(dirname='tests/structures', outname=self.outname)
        prepare_data(source=self.outname, split_ratio=self.split_ratio)

    def test_overlap_and_ratio(self):
        names = np.load(self.outname).files
        
        train_names = get_names(os.path.join(self.tempdir.name, 'train.json'))
        val_names = get_names(os.path.join(self.tempdir.name, 'validation.json'))
        test_names = get_names(os.path.join(self.tempdir.name, 'test.json'))

        # Their pairwise intersections must be the empty set.
        for comb in combinations([train_names, val_names, test_names], r=2):
            self.assertEqual(set(comb[0]) & set(comb[1]), set())

        # Their sizes must be equal to split_ratio.
        self.assertEqual(
                (len(train_names), len(val_names), len(test_names)),
                self.split_ratio
                )

        # Their union must equal names.
        self.assertEqual(
                set(train_names) | set(val_names) | set(test_names),
                set(names)
                )

    def tearDown(self):
        self.tempdir.cleanup()


class TestPCDDataset(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        self.outname = os.path.join(self.tempdir.name, 'pcds.npz')

        pcd_from_dir(dirname='tests/structures', outname=self.outname)

        _npz = np.load(self.outname)
        self.pcd_names = _npz.files

        self.path_to_X = self.outname
        self.path_to_Y = 'tests/dummy/toy_dataset.csv'
        self.index_col = 'id'
        self.labels = ['y1', 'y2']
        self.transform_x = Center()
        self.transform_y = lambda y: y - 1
        self.batch_size = 2

    def test_labeled_pcddataset(self):
        X = np.load(self.outname)
        Y = pd.read_csv(self.path_to_Y, index_col=self.index_col)[self.labels]

        dataset = PCDDataset(
                pcd_names=self.pcd_names,
                path_to_X=self.path_to_X,
                path_to_Y=self.path_to_Y,
                index_col=self.index_col,
                labels=self.labels,
                transform_x=self.transform_x,
                transform_y=self.transform_y
                )

        # Check the size of the dataset.
        self.assertEqual(len(dataset), len(self.pcd_names))
        
        for i in range(len(dataset)):
            name = self.pcd_names[i]

            # Untransformed sample.
            x = torch.tensor(X[name])
            y = torch.tensor(Y.loc[name].to_numpy())

            # Transformed sample.
            sample_x, sample_y = dataset[i]

            # Check that transformations are applied.
            self.assertFalse(torch.equal(sample_x, x))
            self.assertFalse(torch.equal(sample_y, y))

            # Check that self.transform_x is correctly applied.
            self.assertTrue(torch.allclose(sample_x.mean(axis=0)[:3], torch.zeros(3), atol=1e-4))

            # Check that self.transform_y is correctly applied.
            self.assertTrue(torch.equal(y - 1, sample_y))

        # Check that it works properly with a dataloader.
        for x, y in DataLoader(
                dataset, batch_size=self.batch_size,
                collate_fn=Collator(),
                num_workers=1,
                ):
            self.assertEqual(x.ndim, 3)
            self.assertEqual(len(x), self.batch_size)
            self.assertEqual(x.dtype, torch.float)
            self.assertEqual(y.shape, (self.batch_size, len(self.labels)))
            self.assertEqual(y.dtype, torch.float)

    def test_unlabeled_pcddataset(self):
        X = np.load(self.outname)

        dataset = PCDDataset(
                pcd_names=self.pcd_names,
                path_to_X=self.path_to_X,
                transform_x=None,
                )

        # Check the size of the dataset.
        self.assertEqual(len(dataset), len(self.pcd_names))
        
        for i in range(len(dataset)):
            name = self.pcd_names[i]

            # Untransformed sample.
            x = torch.tensor(X[name])

            # "Transformed" sample.
            sample_x = dataset[i]

            # Check that transformations are not applied.
            self.assertTrue(torch.equal(sample_x, x))

        # Check that it works properly with a dataloader.
        for x in DataLoader(
                dataset, batch_size=self.batch_size,
                collate_fn=pad_pcds,
                num_workers=1,
                ):
            self.assertEqual(len(x), self.batch_size)
            self.assertEqual(x.ndim, 3)
            self.assertEqual(x.dtype, torch.float)

    def tearDown(self):
        self.tempdir.cleanup()


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(data))
    return tests
