r"""
Fill this later.
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
