# This file is part of AIdsorb.
# Copyright (C) 2024 Antonios P. Sarikas

# MOXελ is free software: you can redistribute it and/or modify
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
Unit tests for the aidsorb._cli module.

Run from: project's root directory
Command: python -m unittest tests.test__cli
"""

import os
import tempfile
import unittest


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        self.outname = os.path.join(self.tempdir.name, 'pcds.npz')
        self.dirname = 'tests/samples'

    def test_cli(self):
        os.system(f'aidsorb create {self.dirname} {self.outname}')
        os.system(f'aidsorb prepare {self.outname}')

        # Check that the files are correctly created.
        self.assertTrue(os.path.isfile(self.outname))
        for mode in ['train', 'validation', 'test']:
            self.assertTrue(os.path.isfile(f'{self.tempdir.name}/{mode}.json'))

        # Check that LightningCLI works.
        os.system(f'aidsorb-lit fit \
                --config=tests/dummy/config_example.yaml \
                --data.path_to_X={self.outname}'
                  )

    def tearDown(self):
        self.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
