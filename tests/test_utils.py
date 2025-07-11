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
Unit tests for the aidsorb.utils module.

Run from: project's root directory
Command: python -m unittest tests.test_utils
"""

import doctest
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aidsorb import utils
from aidsorb.utils import pcd_from_dir, pcd_from_file, pcd_from_files


class TestPCDFromFile(unittest.TestCase):
    def test_shape(self):
        name, pcd = pcd_from_file('tests/structures/IRMOF-1.xyz')
        self.assertEqual(name, 'IRMOF-1')
        self.assertEqual(pcd.shape, (424, 4))

    def test_features(self):
        # Features: Pauling electronegativity and period number.
        water = np.array([
            [0, 0, 0.11779, 8, 3.44, 2],
            [0, 0.75545, -0.47116, 1, 2.20, 1],
            [0, -0.75545, -0.47116, 1, 2.20, 1]],
            dtype='float32'
            )

        name, pcd = pcd_from_file(
                'tests/dummy/H2O.xyz',
                features=['en_pauling', 'period'],
                )

        self.assertEqual(name, 'H2O')
        self.assertTrue(np.array_equal(pcd, water))


class TestPCDFromFiles(unittest.TestCase):
    def setUp(self):
        # The test assumes all files are processable.
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        self.outname = os.path.join(self.tempdir.name, 'pcd_data')
        self.filenames = ['tests/structures/IRMOF-1.xyz', 'tests/structures/Cu-BTC.cif']
        self.names = [Path(i).stem for i in self.filenames]

    def test_pcd_from_files(self):
        pcd_from_files(self.filenames, outname=self.outname)
        
        # Load the .npy files stored under <outname> directory.
        data = {}
        for name in self.names:
            path_to_npy = os.path.join(self.outname, f'{name}.npy')
            data[name] = np.load(path_to_npy)

        # Point cloud of IRMOF-1 should include Zinc (Z=30).
        self.assertTrue(30 in data['IRMOF-1'][:, -1])

        # Point cloud of Cu-BTC should include Copper (Z=29).
        self.assertTrue(29 in data['Cu-BTC'][:, -1])

        # Check that pcds have the correct shape.
        self.assertEqual(data['IRMOF-1'].shape, (424, 4))
        self.assertEqual(data['Cu-BTC'].shape, (624, 4))

    def tearDown(self):
        self.tempdir.cleanup()


class TestPCDFromDir(unittest.TestCase):
    def setUp(self):
        # The test assumes all files are processable.
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        self.outname = os.path.join(self.tempdir.name, 'pcd_data')
        self.dirname = 'tests/structures'

    def test_pcd_from_dir(self):
        pcd_from_dir(dirname=self.dirname, outname=self.outname)

        # Check the number of converted files.
        self.assertEqual(len(os.listdir(self.dirname)), len(os.listdir(self.outname)))

    def tearDown(self):
        self.tempdir.cleanup()


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(utils))
    return tests
