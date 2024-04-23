import os
from pathlib import Path
import doctest
import unittest
import tempfile
import numpy as np
from aidsorb import utils
from aidsorb.utils import pcd_from_file, pcd_from_files, pcd_from_dir


class TestPCDFromFile(unittest.TestCase):
    def test_shape(self):
        name, pcd = pcd_from_file('tests/samples/IRMOF-1.xyz')
        self.assertEqual(name, 'IRMOF-1')
        self.assertEqual(pcd.shape, (424, 4))

    def test_features(self):
        # Features: Pauling electronegativity and period.
        water = np.array([
            [0, 0, 0.11779, 8, 3.44, 2],
            [0, 0.75545, -0.47116, 1, 2.20, 1],
            [0, -0.75545, -0.47116, 1, 2.20, 1],
            ], dtype='float32')

        name, pcd = pcd_from_file(
                'tests/dummy/H2O.xyz',
                features=['en_pauling', 'period'],
                )

        self.assertEqual(name, 'H2O')
        self.assertTrue(np.all(pcd == water))


class TestPCDFromFiles(unittest.TestCase):
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


class TestPCDFromDir(unittest.TestCase):
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


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(utils))
    return tests


if __name__ == '__main__':
    unittest.main()
