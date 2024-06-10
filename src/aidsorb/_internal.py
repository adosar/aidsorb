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

from mendeleev.fetch import fetch_table


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

# Load the periodic table.
_ptable = fetch_table('elements')
_ptable.set_index('atomic_number', inplace=True)
