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
        self.outname = os.path.join(self.tempdir.name, 'pcd_data')
        self.dirname = 'tests/structures'
        self.split_ratio = [2, 2, 2]
        self.path_to_logs = f'{self.tempdir.name}/lightning_logs/version_0'

    def run_command(self, command):
        exit_status = os.system(command)
        if exit_status != 0:
            raise RuntimeError(command)

    def test_cli(self):
        self.run_command(f'aidsorb create {self.dirname} {self.outname}')
        self.run_command(f'aidsorb prepare {self.outname} --split_ratio "{self.split_ratio}"')

        # Check that the files are correctly created.
        self.assertTrue(os.path.isdir(self.outname))
        for mode in ['train', 'validation', 'test']:
            self.assertTrue(os.path.isfile(f'{self.tempdir.name}/{mode}.json'))

        # Check that LightningCLI works.
        self.run_command(
                f'aidsorb-lit fit'
                f' --config=tests/dummy/config_example.yaml'
                f' --data.path_to_X={self.outname}'
                f' --trainer.default_root_dir={self.tempdir.name}'
                )

        for mode in ['validate', 'test']:
            self.run_command(
                    f'aidsorb-lit {mode}'
                    f' --config={self.path_to_logs}/config.yaml'
                    f' --ckpt_path={self.path_to_logs}/checkpoints/best.ckpt'
                    )

    def tearDown(self):
        self.tempdir.cleanup()
