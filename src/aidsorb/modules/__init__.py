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
:class:`torch.nn.Module`'s for 3D deep learning.

.. rubric:: Submodules

.. autosummary::
   :nosignatures:
   :template: module.rst
   :toctree:

   points
   voxels

"""

from .points import (
        PointNet,
        PointNetBackbone,
        PointNetClsHead,
        PointNetSegHead,
        )

from .voxels import (
        RetNeXt,
        IntelliPore,
        )
