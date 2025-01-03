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
Helper functions and classes for transforming point clouds.

.. note::
    ``pcd`` must be a :class:`~torch.Tensor` of shape ``(N, 3+C)``.

.. warning::
    For efficiency reasons, **transformations do not copy input tensors**.

.. tip::
    For implementing your own transforms, have a look at the transforms
    `tutorial`_. For more flexibility, consider implementing them as callable
    instances of classes.

    .. _tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
"""

import torch
from roma import random_rotmat
from . _internal import check_shape


def upsample_pcd(pcd, size):
    r"""
    Upsample ``pcd`` to a new ``size`` by sampling with replacement from ``pcd``.

    Parameters
    ----------
    pcd : tensor of shape (N, C)
        Original point cloud of size ``N``.
    size : int
        Size of the new point cloud.

    Returns
    -------
    new_pcd : tensor of shape (size, C)

    Examples
    --------
    >>> pcd = torch.tensor([[2, 4, 5, 6]])
    >>> upsample_pcd(pcd, 3)
    tensor([[2, 4, 5, 6],
            [2, 4, 5, 6],
            [2, 4, 5, 6]])

    >>> # New points point must be from pcd.
    >>> pcd = torch.randn(10, 4)
    >>> new_pcd = upsample_pcd(pcd, 20)
    >>> (new_pcd[-1] == pcd).all(1).any()  # Check for last point.
    tensor(True)

    >>> # No upsampling.
    >>> pcd = torch.randn(100, 4)
    >>> new_pcd = upsample_pcd(pcd, len(pcd))
    >>> torch.equal(pcd, new_pcd)
    True

    """
    n_samples = size - len(pcd)
    indices = torch.randint(len(pcd), (n_samples,))  # With replacement.
    new_points = pcd[indices]

    return torch.cat((pcd, new_points))


def split_pcd(pcd):
    r"""
    Split a point cloud to coordinates and features.

    Parameters
    ----------
    pcd : tensor of shape (N, 3+C)

    Returns
    -------
    coords_feats : tuple of length 2
        * ``coords_and_feats[0] == coords`` tensor of shape (N, 3).
        * ``coords_and_feats[1] == feats`` tensor of shape (N, C).

    Examples
    --------
    >>> pcd = torch.randn(25, 7)  # Point cloud with 4 features.
    >>> coords, feats = split_pcd(pcd)
    >>> coords.shape
    torch.Size([25, 3])
    >>> feats.shape
    torch.Size([25, 4])

    >>> pcd = torch.randn(15, 3)  # Point cloud with no features.
    >>> coords, feats = split_pcd(pcd)
    >>> coords.shape
    torch.Size([15, 3])
    >>> feats.shape
    torch.Size([15, 0])
    """
    check_shape(pcd)

    return pcd[:, :3], pcd[:, 3:]


def transform_pcd(pcd, tfm):
    r"""
    Transform the coordinates of a point cloud.

    For molecular point clouds, *only rigid transformations are recommended*.

    Parameters
    ----------
    pcd : tensor of shape (N, 3+C)
        Original point cloud.

    tfm : tensor of shape (3, 3)
        Transformation matrix.

    Returns
    -------
    new_pcd : tensor of shape (N, 3+C)
        Transformed point cloud.

    Raises
    ------
    ValueError
        If ``pcd`` or ``tfm`` do not have the expected shape.

    Examples
    --------
    >>> pcd = torch.tensor([[3, -9, 2, 6], [3, 4, -1, 8]])
    >>> tfm = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> transform_pcd(pcd, tfm)
    tensor([[ 9,  3,  2,  6],
            [-4,  3, -1,  8]])

    >>> pcd = torch.randn(424, 2)  # Invalid shape.
    >>> transform_pcd(pcd, tfm)
    Traceback (most recent call last):
        ...
    ValueError: Expecting shape (N, 3+C) but got shape (424, 2)!
    """
    check_shape(pcd)

    if not tfm.shape == (3, 3):
        raise ValueError(
                'Expecting array of shape (3, 3) '
                f'but got array of shape {tfm.shape}!'
                )

    coords, feats = split_pcd(pcd)
    new_coords = coords @ tfm.T  # Transpose the matrix.

    return torch.hstack((new_coords, feats))


def center_pcd(pcd):
    r"""
    Center the coordinates of a point cloud by subtracting their centroid.

    Parameters
    ----------
    pcd : tensor of shape (N, 3+C)

    Returns
    -------
    new_pcd : tensor of shape (N, 3+C)
        Centered point cloud.

    Raises
    ------
    ValueError
        If ``pcd`` does not have the expected shape.

    Examples
    --------
    >>> pcd = torch.tensor([[2., 1., 3., 6.], [-3., 2., 8., 8.]])
    >>> new_pcd = center_pcd(pcd)
    >>> new_pcd.mean(dim=0)
    tensor([0., 0., 0., 7.])

    >>> pcd = torch.randn(5, 2)  # Invalid shape.
    >>> new_pcd = center_pcd(pcd)
    Traceback (most recent call last):
        ...
    ValueError: Expecting shape (N, 3+C) but got shape (5, 2)!
    """
    check_shape(pcd)

    centroid = pcd.mean(dim=0)
    centroid[3:] = 0  # Center only the coordinates.

    return pcd - centroid


class Center():
    r"""
    Center the coordinates of a point cloud by subtracting their centroid.

    See also
    --------
    :func:`.center_pcd`
        For a functional interface.
    Examples
    --------
    >>> x = torch.arange(4.)
    >>> pcd = torch.stack((x, x))
    >>> center = Center()
    >>> center(pcd)
    tensor([[0., 0., 0., 3.],
            [0., 0., 0., 3.]])
    """
    def __call__(self, pcd):
        return center_pcd(pcd)  # Checks also for shape.


class RandomRotation():
    r"""
    Randomly rotate the coordinates of a point cloud.

    Examples
    --------
    >>> pcd = torch.randn(25, 4)
    >>> rot = RandomRotation()
    >>> new_pcd = rot(pcd)
    >>> new_pcd.shape
    torch.Size([25, 4])

    >>> coords, feats = split_pcd(pcd)
    >>> new_coords, new_feats = split_pcd(new_pcd)

    >>> torch.equal(new_coords, coords)  # Coordinates are affected.
    False
    >>> torch.equal(new_feats, feats)  # Features are not affected.
    True
    """
    def __call__(self, pcd):
        check_shape(pcd)
        rr = random_rotmat()

        return transform_pcd(pcd=pcd, tfm=rr)


class Jitter():
    r"""
    Jitter the coordinates of a point cloud by adding zero-mean normal noise.

    Parameters
    ----------
    std: float
        Standard deviation of the normal noise.

    Examples
    --------
    >>> pcd = torch.randn(100, 5)
    >>> jitter = Jitter(0.01)
    >>> new_pcd = jitter(pcd)
    >>> new_pcd.shape
    torch.Size([100, 5])

    >>> coords, feats = split_pcd(pcd)
    >>> new_coords, new_feats = split_pcd(new_pcd)

    >>> torch.equal(new_coords, coords)  # Coordinates are affected.
    False
    >>> torch.equal(new_feats, feats)  # Features are not affected.
    True
    """
    def __init__(self, std):
        self.std = std

    def __call__(self, pcd):
        check_shape(pcd)

        noise = torch.normal(mean=0, std=self.std, size=pcd.shape)
        noise[:, 3:] = 0  # Jitter only the coordinates.

        return pcd + noise


class RandomErase():
    r"""
    Randomly erase a number of points from the point cloud.

    .. todo::
        Consider adding the option for a fraction of points to be erased.

    Parameters
    ----------
    n_points : int, default=5
        Number of points to be erased.

    Examples
    --------
    >>> pcd = torch.randn(100, 5)
    >>> erase = RandomErase(n_points=10)
    >>> erase(pcd).shape
    torch.Size([90, 5])
    """
    def __init__(self, n_points):
        self.n_points = n_points

    def __call__(self, pcd):
        check_shape(pcd)

        # Indices of points to keep.
        keep_size = len(pcd) - self.n_points
        indices = torch.randperm(len(pcd))[:keep_size]

        return pcd[indices]
