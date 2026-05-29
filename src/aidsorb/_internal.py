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

r"""Helper functions and data for use in other modules."""

from importlib.resources import files

import pandas as pd

# This will be the default on Pandas 3.0
pd.options.mode.copy_on_write = True

# Load the periodic table.
with files('aidsorb.pkg_data').joinpath('periodic_table.csv').open() as fhand:
    ptable = pd.read_csv(fhand, index_col='atomic_number')
