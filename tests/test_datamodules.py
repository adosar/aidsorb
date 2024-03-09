import unittest
import tempfile
from aidsorb.utils import pcd_from_dir
from aidsorb.data import prepare_data


class TestPCDDataModule(unittest.TestCase):
    def setUp(self):
        ...
        #self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        #self.outname = os.path.join(self.tempdir.name, 'pcds.npz')
        #self.split_ratio = [2, 1, 3]

        #pcd_from_dir(dirname='tests/samples', outname=self.outname)
        #prepare_data(source=self.outname, split_ratio=self.split_ratio)

    def test_prepare_data(self):
        ...

    def tearDown(self):
        ...
        #self.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
