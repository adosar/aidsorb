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
from aidsorb._internal import _check_shape
from aidsorb.utils import pcd_from_dir
from aidsorb.transforms import Centering
from aidsorb.data import (
        get_names, prepare_data, PCDDataset,
        zero_pad_pcds, collate_zero_pad_pointnet
        )


class TestPrepareData(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        self.outname = os.path.join(self.tempdir.name, 'pcds.npz')
        self.split_ratio = (2, 1, 3)

        pcd_from_dir(dirname='tests/samples', outname=self.outname)
        prepare_data(source=self.outname, split_ratio=self.split_ratio)

    def test_overlap_and_ratio(self):
        names = np.load(self.outname, mmap_mode='r').files
        
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

        pcd_from_dir(dirname='tests/samples', outname=self.outname)

        self.pcd_names = np.load(self.outname).files
        self.path_to_X = self.outname
        self.path_to_Y = 'tests/samples.csv'
        self.index_col = 'id'
        self.labels = ['y1', 'y2']
        self.transform_x = Centering()
        self.transform_y = lambda y: y - 1  # Decrease all outputs by 1.
        self.batch_size = 2

        self.X = np.load(self.outname, mmap_mode='r')
        self.Y = pd.read_csv(self.path_to_Y, index_col=self.index_col)[self.labels]

    #@unittest.skip
    def test_labeled_pcddataset(self):
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
            x = torch.tensor(self.X[name])
            y = torch.tensor(self.Y.loc[name].values)

            # Transformed sample.
            sample_x, sample_y = dataset[i]

            # Check that transformations are applied.
            self.assertFalse(torch.all(sample_x == x))
            self.assertFalse(torch.all(sample_y == y))

            # Check that self.transform_x is correctly applied.
            self.assertTrue(torch.all(sample_x.mean(axis=0)[:3].int() == 0))

            # Check that self.transform_y is correctly applied.
            self.assertTrue(torch.all(y - 1 == sample_y))

            # Check that sample has correct dimensions/shape.
            self.assertEqual(_check_shape(sample_x), None)
            self.assertEqual(sample_y.shape, (len(self.labels),))

        # Check that it works properly with a dataloader.
        for x, y in DataLoader(
                dataset, batch_size=self.batch_size,
                collate_fn=collate_zero_pad_pointnet,
                num_workers=2,
                ):
            self.assertEqual(x.ndim, 3)
            self.assertEqual(len(x), self.batch_size)
            self.assertEqual(x.dtype, torch.float)
            self.assertEqual(y.shape, (self.batch_size, len(self.labels)))
            self.assertEqual(y.dtype, torch.float)

    def test_unlabeled_pcddataset(self):
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
            x = torch.tensor(self.X[name])

            # "Transformed" sample.
            sample_x = dataset[i]

            # Check that transformations are not applied.
            self.assertTrue(torch.all(sample_x == x))

            # Check that sample has correct dimensions/shape.
            self.assertEqual(_check_shape(sample_x), None)

        # Check that it works properly with a dataloader.
        for x in DataLoader(
                dataset, batch_size=self.batch_size,
                collate_fn=lambda b: zero_pad_pcds(b, channels_first=True),
                num_workers=2,
                ):
            self.assertEqual(len(x), self.batch_size)
            self.assertEqual(x.ndim, 3)
            self.assertEqual(x.dtype, torch.float)

    def tearDown(self):
        self.tempdir.cleanup()


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(data))
    return tests


if __name__ == '__main__':
    unittest.main()
