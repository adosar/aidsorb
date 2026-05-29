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
Helper functions and classes for transforming voxels.

.. note::
    All geometric transforms expect an input :class:`~torch.Tensor` of shape ``(C, D, H, W)``.
"""

from itertools import combinations

import torch


def _check_shape(obj):
    r"""
    Check if ``obj`` has valid shape to be considered voxels.

    Parameters
    ----------
    obj : array or tensor

    Examples
    --------
    >>> x = torch.randn(2, 2, 2, 2)
    >>> _check_shape(x)
    >>> x = torch.randn(2, 2)
    >>> _check_shape(x)
    Traceback (most recent call last):
    ...
    ValueError: expecting shape (C, D, H, W) but received shape (2, 2)

    Raises
    ------
    ValueError
        If ``obj.shape != (N, 3+C)``.
    """
    if not (obj.ndim == 4):
        raise ValueError(
                'expecting shape (C, D, H, W) '
                f'but received shape {tuple(obj.shape)}'
                )


class AddChannelDim:
    r"""
    Prepend a dimension to the input tensor.

    Examples
    --------
    >>> x = torch.randn(32, 32, 32)
    >>> AddChannelDim()(x).shape
    torch.Size([1, 32, 32, 32])
    """
    def __call__(self, x):
        return x.unsqueeze(0)


class BoltzmannFactor:
    r"""
    Fill voxels with the Boltzmann factor.

    Parameters
    ----------
    temperature : float

    Examples
    --------
    >>> x = torch.tensor([0., torch.inf])
    >>> BoltzmannFactor()(x)
    tensor([1., 0.])
    """
    def __init__(self, temperature: float = 298.):
        self.temperature = temperature

    def __call__(self, x):
        return torch.exp((-1 / self.temperature) * x)


class ClipVoxels:
    r"""
    Clip voxels within ``[vmin, vmax]``.

    Parameters
    ----------
    vmin : float
    vmax : float

    Examples
    --------
    >>> x = torch.tensor([-20., 22.])
    >>> out = ClipVoxels(-1, 1)(x)
    >>> out
    tensor([-1.,  1.])
    """
    def __init__(self, vmin: float, vmax: float):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, x):
        return torch.clip(x, self.vmin, self.vmax)


class ClipScaleVoxels:
    r"""
    Clip and then normalize voxels within ``[-1, 1]``.

    First clips voxels within ``[-value, value]``, then divides the result by
    ``value``, producing voxels with values in ``[-1, 1]``.

    Parameters
    ----------
    value : float

    Examples
    --------
    >>> x = torch.tensor([-12., 11.])
    >>> ClipScaleVoxels(10)(x)
    tensor([-1.,  1.])
    """
    def __init__(self, value: float = 5e3):
        self.value = abs(value)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_clipped = torch.clamp(x, -self.value, self.value)
        x_scaled = x_clipped / self.value  # Scale to [-1, 1]

        return x_scaled


class RandomNoise:
    r"""
    Add normal noise to voxels.

    Parameters
    ----------
    std : float
        Standard deviation of the normal noise.

    Examples
    --------
    >>> x = torch.randn(3, 3)
    >>> out = RandomNoise(0.1)(x)
    >>> out.shape
    torch.Size([3, 3])
    >>> torch.equal(x, out)
    False
    """
    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        noise = torch.randn(x.shape, device=x.device) * self.std
        return x + noise


class RandomRotation90:
    r"""
    Rotate voxels around a randomly chosen axis by 90 degrees.

    Examples
    --------
    >>> x = torch.randn(2, 3, 3, 3)
    >>> out = RandomRotation90()(x)
    >>> out.shape
    torch.Size([2, 3, 3, 3])
    >>> torch.equal(x, out)
    False
    """
    def __init__(self):
        self.planes = list(combinations([1, 2, 3], 2))
        self.directions = torch.tensor([-1, 1])

    def __call__(self, x):
        _check_shape(x)
        p_choice = torch.randint(len(self.planes), ()).item()
        plane = self.planes[p_choice]

        d_choice = torch.randint(len(self.directions), ()).item()
        direction = self.directions[d_choice]

        return torch.rot90(x, k=direction, dims=plane)


class RandomFlip:
    r"""
    Flip voxels along a randomly chosen axis.

    Examples
    --------
    >>> x = torch.randn(2, 3, 3, 3)
    >>> out = RandomFlip()(x)
    >>> out.shape
    torch.Size([2, 3, 3, 3])
    >>> torch.equal(x, out)
    False
    """
    def __call__(self, x):
        _check_shape(x)
        dim = torch.randint(1, 4, ()).item()

        return torch.flip(x, [dim])


class RandomReflect:
    r"""
    Reflect voxels along a randomly chosen plane.

    Examples
    --------
    >>> x = torch.randn(2, 3, 3, 3)
    >>> out = RandomReflect()(x)
    >>> out.shape
    torch.Size([2, 3, 3, 3])
    >>> torch.equal(x, out)
    False
    """
    def __init__(self):
        self.planes = list(combinations([1, 2, 3], 2))

    def __call__(self, x):
        _check_shape(x)
        p_choice = torch.randint(len(self.planes), ()).item()
        plane = self.planes[p_choice]

        return torch.transpose(x, *plane)
