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
This module provides helper functions and data for use in other modules.
"""

from importlib.resources import files
import pandas as pd


def _check_shape(array):
    r"""
    Check if ``array`` has valid shape to be considered a point cloud.

    Parameters
    ----------
    array

    Raises
    ------
    ValueError
        If ``array.shape != (N, 3+C)``.
    """
    if not ((array.ndim == 2) and (array.shape[1] >= 3)):
        raise ValueError(
                'Expecting array of shape (N, 3+C) '
                f'but got array of shape {array.shape}!'
                )


def _check_shape_vis(array):
    r"""
    Check if ``array`` has valid shape to be considered a molecular point cloud.

    Parameters
    ----------
    array

    Raises
    ------
    ValueError
        If ``array.shape != (N, 4+C)``.
    """
    if not ((array.ndim == 2) and (array.shape[1] >= 4)):
        raise ValueError(
                'Expecting array of shape (N, 4+C) '
                f'but got array of shape {array.shape}!'
                )


# Default value for controlling randomness.
_SEED = 1

# This will be the default on Pandas 3.0
pd.options.mode.copy_on_write = True

# Load the periodic table.
with files('aidsorb.pkg_data').joinpath('periodic_table.csv').open() as fhand:
    _ptable = pd.read_csv(fhand, index_col='atomic_number')
