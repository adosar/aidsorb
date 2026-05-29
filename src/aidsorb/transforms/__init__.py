# This file is part of AIdsorb.
# Copyright (C) 2026 Antonios P. Sarikas

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
Helper functions and classes for transforming inputs.

.. warning::
    Transforms avoid in-place modifications. However, **the output tensor(s)
    might be view(s) of the input tensor**. If it is necessary to preserve the
    original data, it is recommended to copy them before applying the
    transform.

.. note::
    All transforms are implemented using :mod:`torch`. Any randomness is handled
    through PyTorch's RNG, so reproducibility can be controlled with
    :func:`torch.manual_seed`.

.. tip::
    For implementing your own transforms, have a look at the transforms
    `tutorial`_. For more flexibility, consider implementing them as callable
    instances of classes. **If your transforms use some source of randomness, it
    is recommended to control it with** :mod:`torch`.

    .. _tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms

.. rubric:: Submodules

.. autosummary::
   :nosignatures:
   :template: module.rst
   :toctree:

   points
   voxels

"""
