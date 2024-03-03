import os
from pathlib import Path
import unittest
import tempfile
import numpy as np
from aidsorb.utils import (
        pcd_from_file, pcd_from_files, pcd_from_dir,
        prepare_data, get_names,
        )


class TestPointCloudFromFile(unittest.TestCase):
    def test_pcd_from_file(self):
        name, pcd = pcd_from_file('tests/samples/IRMOF-1.xyz')

        self.assertEqual(name, 'IRMOF-1')
        self.assertEqual(pcd.shape, (424, 4))


class TestPointCloudFromFiles(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        self.fnames = ['tests/samples/IRMOF-1.xyz', 'tests/samples/Cu-BTC.cif']
        self.outname = os.path.join(self.tempdir.name, 'pcds.npz')
        self.names = [Path(i).stem for i in self.fnames]

    def test_pcd_from_files(self):
        for shuffle in [True, False]:
            with self.subTest():
                pcd_from_files(self.fnames, outname=self.outname, shuffle=shuffle)
                data = np.load(self.outname, mmap_mode='r')

                self.assertEqual(len(data.files), len(self.names))

                if shuffle:
                    # Stored names must include self.names.
                    self.assertEqual(set(data.files), set(self.names))
                else:
                    # Stored names must follow the order in self.names.
                    self.assertEqual(data.files, self.names)

                # Point cloud of IRMOF-1 should include Zirconium (Z=30).
                self.assertTrue(30 in data['IRMOF-1'][:, -1])

                # Point cloud of Cu-BTC should not include Zirconium (Z=30).
                self.assertFalse(30 in data['Cu-BTC'][:, -1])

                # Check that pcds have the correct shape.
                self.assertEqual(data['IRMOF-1'].shape, (424, 4))
                self.assertEqual(data['Cu-BTC'].shape, (624, 4))

    def tearDown(self):
        self.tempdir.cleanup()


class TestPointCloudFromDir(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        self.outname = os.path.join(self.tempdir.name, 'pcds.npz')
        self.dirname = 'tests/samples'
        self.names = [Path(i).stem for i in os.listdir(self.dirname)]

    def test_pcd_from_dir(self):
        for shuffle in [True, False]:
            with self.subTest():
                pcd_from_dir(dirname=self.dirname, outname=self.outname, shuffle=shuffle)
                data = np.load(self.outname, mmap_mode='r')

                self.assertEqual(len(data.files), len(self.names))

                if shuffle:
                    # Stored names must equal self.names.
                    self.assertEqual(set(data.files), set(self.names))
                else:
                    # Stored names must follow the order in self.names.
                    self.assertEqual(data.files, self.names)

                self.assertEqual(data['IRMOF-1'].shape, (424, 4))
                self.assertEqual(data['Cu-BTC'].shape, (624, 4))

    def tearDown(self):
        self.tempdir.cleanup()


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


if __name__ == '__main__':
    unittest.main()
