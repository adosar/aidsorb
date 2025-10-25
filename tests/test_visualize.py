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
Unit tests for the aidsorb.visualize module.

Run from: project's root directory
Command: python -m unittest tests.test_visualize
"""

import doctest
import unittest

import numpy as np
from aidsorb import visualize


class TestDrawPCDFromFile(unittest.TestCase):
    def test_draw_random_pcd(self):
        pcd = np.random.randn(20, 3)
        fig = visualize.draw_pcd(pcd, feature_to_color=(0, 'X'), molecular=False, size=4.5)
        fig.show()
    def test_draw_from_structure(self):
        visualize.draw_pcd_from_file('tests/structures/IRMOF-1.xyz')
    def test_draw_from_npy(self):
        visualize.draw_pcd_from_file('tests/dummy/toy_project/pcd_data/ZnMOF-74.npy')


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(visualize))
    return tests
