import os
import unittest
import tempfile
from itertools import combinations, product
from torch.utils.data import RandomSampler, SequentialSampler
from aidsorb.utils import pcd_from_dir
from aidsorb.data import prepare_data, Collator
from aidsorb.transforms import Centering, RandomRotation
from aidsorb.datamodules import PCDDataModule


def _dummy_trans_y(y):
    return y - 1


class TestPCDDataModule(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir='/tmp')
        self.outname = os.path.join(self.tempdir.name, 'pcds.npz')
        self.split_ratio = [3, 1, 2]

        pcd_from_dir(dirname='tests/samples', outname=self.outname)
        prepare_data(source=self.outname, split_ratio=self.split_ratio)

        # Arguments for the datamodule.
        self.train_size = 2
        self.train_trans_x = Centering()
        self.eval_trans_x = RandomRotation()
        self.trans_y = _dummy_trans_y
        self.shuffle = True
        self.train_bs = 3
        self.eval_bs = 2
        self.config_dataloaders = {
                'pin_memory': True,
                'num_workers': 2,
                'collate_fn': Collator()
                }

        # Instantiate the datamodule.
        self.dm = PCDDataModule(
                path_to_X=self.outname,
                path_to_Y='tests/samples.csv',
                index_col='id',
                labels=['y2', 'y3'],
                train_size=self.train_size,
                train_transform_x=self.train_trans_x,
                eval_transform_x=self.eval_trans_x,
                transform_y=self.trans_y,
                shuffle=self.shuffle,
                train_batch_size=self.train_bs,
                eval_batch_size=self.eval_bs,
                config_dataloaders=self.config_dataloaders,
                )

        self.dm.prepare_data()
        self.dm.setup()

    def test_datasets(self):
        # Check that the datasets have the correct size.
        self.assertEqual(len(self.dm.train_dataset), self.train_size)
        self.assertEqual(len(self.dm.validation_dataset), self.split_ratio[1])
        self.assertEqual(len(self.dm.test_dataset), self.split_ratio[2])

        # The pairwise intersections must be the empty set.
        for ds_comb in combinations([
                self.dm.train_dataset.pcd_names,
                self.dm.validation_dataset.pcd_names,
                self.dm.test_dataset.pcd_names,
                ], r=2):
            self.assertEqual(set(ds_comb[0]) & set(ds_comb[1]), set())

        # Check that the transformations are passed correcly.
        for ds in (
                self.dm.train_dataset,
                self.dm.validation_dataset,
                self.dm.test_dataset,
                ):
            if ds is self.dm.train_dataset:
                self.assertIs(ds.transform_x, self.train_trans_x)
            else:
                self.assertIs(ds.transform_x, self.eval_trans_x)

            self.assertIs(ds.transform_y, self.trans_y)


    def test_dataloaders(self):
        dataloaders = [
                self.dm.train_dataloader(),
                self.dm.val_dataloader(),
                self.dm.test_dataloader(),
                ]
        datasets = [
                self.dm.train_dataset,
                self.dm.validation_dataset,
                self.dm.test_dataset,
                ]

        passed_collate_fn = self.config_dataloaders['collate_fn']

        for i, dl in enumerate(dataloaders):
            # Check that dataloaders use appropriate settings.
            if i == 0:
                self.assertIsInstance(dl.sampler, RandomSampler)
                self.assertEqual(dl.batch_size, self.train_bs)
            else:
                self.assertIsInstance(dl.sampler, SequentialSampler)
                self.assertEqual(dl.batch_size, self.eval_bs)

            self.assertEqual(dl.collate_fn, passed_collate_fn)

            # Check that collate function is used properly.
            for x, y in dl:
                self.assertEqual(x.ndim, 3)
                self.assertEqual(y.ndim, 2)

        # Check that each dataloaders loads the correct dataset.
        for dl, ds in zip(dataloaders, datasets):
            self.assertIs(dl.dataset, ds)

    def tearDown(self):
        self.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
