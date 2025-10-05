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

r"""Helper module for :mod:`.transforms`."""

import torch
from torch import Tensor


def points_not_affected(
        pcd: Tensor,
        n_points: int | float,
        return_affected: bool = False
        ) -> int:
    r"""
    Return the number of points not affected by a transform.

    Parameters
    ----------
    pcd : tensor of shape (N, 3+C)
    n_points : int or float
        Number or fraction of points that will be affected. If :class:`float`,
        it shoud be in the interval ``(0, 1)``.
    return_affected : bool, default=False
        Whether to return the number of affected points.

    Returns
    -------
    count : int

    Examples
    --------
    >>> points_not_affected(torch.randn(50, 4), 30)
    20
    >>> points_not_affected(torch.randn(40, 4), 0.4)
    24
    >>> points_not_affected(torch.randn(40, 4), 0.4, True)
    16
    >>> points_not_affected(torch.randn(40, 4), 50)
    Traceback (most recent call last):
        ...
    RuntimeError: resulting point cloud has no points
    """
    if 0 < n_points < 1:
        count = len(pcd) - int(len(pcd) * n_points)
    else:
        count = len(pcd) - n_points

    if count < 1:
        raise RuntimeError('resulting point cloud has no points')

    return len(pcd) - count if return_affected else count


def local_patch_indices(pcd: Tensor, n_points: int | float) -> Tensor:
    r"""
    Return the indices of a local patch.

    Parameters
    ----------
    pcd : tensor of shape (N, 3+C)
    n_points : int or float
        Number of points or fractional size of the local patch. If
        :class:`float`, it shoud be in the interval ``(0, 1)``. In this case,
        the size of the local patch will be ``int(len(pcd) * n_points)``.

    Returns
    -------
    indices : boolean tensor of shape (N,)
        ``True`` indicates that a point is part of the patch.

    Examples
    --------
    >>> pcd = torch.randn(20, 4)
    >>> indices = local_patch_indices(pcd, 0.3)
    >>> indices.shape
    torch.Size([20])
    >>> indices.dtype
    torch.bool
    """
    k = points_not_affected(pcd, n_points, return_affected=True)
    coords = pcd[:, :3]

    # Randomly sample a point from the cloud.
    idx = torch.randint(len(pcd), size=()).item()

    # Compute distance matrix.
    distances = torch.linalg.vector_norm(coords - coords[idx], dim=1)

    indices = torch.zeros(len(pcd), dtype=torch.bool)
    _, patch_indices = torch.topk(distances, k=k, largest=False, sorted=False)
    indices[patch_indices] = True

    return indices
