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
**AIdsorb** is a :fa:`python; fa-fade` Python package for **deep learning on
molecular point clouds**.


.. admonition:: AIdsorb adopts the following conventions

    * A ``pcd`` is represented as a :class:`numpy.ndarray` of shape ``(N, 3+C)``.
    * A molecular ``pcd`` is represented as a :class:`numpy.ndarray` of shape ``(N, 4+C)``
      where ``N`` is the number of atoms, ``pcd[:, :3]`` are the **atomic
      coordinates**, ``pcd[:, 3]`` are the **atomic numbers** and ``pcd[:, 4:]``
      any **additional features**. If ``C == 0``, then the only features are the
      atomic numbers.
"""

__author__ = 'Antonios P. Sarikas'
__copyright__ = 'Copyright (c) 2024 Antonios P. Sarikas'
__license__ = ' GPL-3.0-only'
