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
