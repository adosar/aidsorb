r"""
Add module docstring.
"""

import fire
from lightning.pytorch.cli import LightningCLI
from . visualize import draw_pcd_from_file
from . utils import pcd_from_dir
from . data import prepare_data
from . datamodules import PCDDataModule
from . litmodels import PointNetLit


def lightning_cli():
    r"""
    Run the :class:`LightningCli`.
    """
    LightningCLI(PointNetLit, PCDDataModule)


def aidsorb_fire():
    r"""
    CLI for the ``AIdsorb`` package.
    """
    fire.Fire({
        'visualize': draw_pcd_from_file,
        'create': pcd_from_dir,
        'prepare': prepare_data,
        })
