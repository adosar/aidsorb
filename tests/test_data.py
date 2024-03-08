import os
import unittest
import tempfile
import numpy as np
import pandas as pd
from aidsorb.utils import pcd_from_dir
from aidsorb.data import get_names, prepare_data, PCDDataset
from aidsorb.transforms import Centering
from aidsorb._internal import _check_shape


class TestPrepareData(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        self.outname = os.path.join(self.tempdir.name, 'pcds.npz')
        self.split_ratio = (2, 1, 2)

        pcd_from_dir(dirname='tests/samples', outname=self.outname)
        prepare_data(source=self.outname, split_ratio=self.split_ratio)

    def test_overlap_and_ratio(self):
        names = np.load(self.outname, mmap_mode='r').files
        
        train_names = get_names(os.path.join(self.tempdir.name, 'train.json'))
        val_names = get_names(os.path.join(self.tempdir.name, 'validation.json'))
        test_names = get_names(os.path.join(self.tempdir.name, 'test.json'))

        # Their pairwise intersections must be the empty set.
        self.assertEqual(set(train_names) & set(val_names), set())
        self.assertEqual(set(train_names) & set(test_names), set())
        self.assertEqual(set(val_names) & set(test_names), set())

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

        self.pcd_X = np.load(self.outname, mmap_mode='r')
        self.pcd_names = self.pcd_X.files
        self.pcd_Y = pd.read_csv('tests/samples.csv', index_col='id')
        self.labels = ['y1', 'y2']
        self.transform_x = Centering()
        self.transform_y = lambda y: y - 1  # Decrease all outputs by 1.

    def test_pcddataset(self):
        dataset = PCDDataset(
                pcd_names=self.pcd_names, pcd_X=self.pcd_X,
                pcd_Y=self.pcd_Y, labels=self.labels,
                transform_x=self.transform_x, transform_y=self.transform_y
                )

        # Check the size of the dataset.
        self.assertEqual(len(dataset), len(self.pcd_names))
        
        for i in range(len(dataset)):
            name = self.pcd_names[i]

            # Untransformed sample.
            x, y = dataset.X[name], dataset.Y.loc[name, self.labels]

            # Transformed sample.
            sample_x, sample_y = dataset[i]

            # Check that transformations are applied.
            self.assertFalse(np.all(sample_x == x))
            self.assertFalse(np.all(sample_y == y))

            # Check that self.transform_x is correctly applied.
            self.assertTrue(np.all(sample_x.mean(axis=0)[:3].astype(int) == 0))

            # Check that self.transform_y is correctly applied.
            self.assertTrue(np.all(y - 1 == sample_y))

            # Check that sample has correct dimensions/shape.
            self.assertEqual(_check_shape(sample_x), None)
            self.assertEqual(sample_y.shape, (len(self.labels),))


    def tearDown(self):
        self.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
