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
This module provides helper functions and classes for transforming point clouds.

.. note::
    The ``pcd`` must be a :class:`~numpy.ndarray` and have shape of ``(N, 3+C)``.

.. tip::
    For implementing your own transforms, have a look at the transforms
    `tutorial`_. For more flexibility, consider implementing them as callable
    instances of classes.

    .. _tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from . utils import split_pcd
from . _internal import _check_shape


def transform_pcd(pcd, tfm):
    r"""
    Transform the coordinates of a point cloud.

    For molecular point clouds, *only rigid transformations are recommended*.

    Parameters
    ----------
    pcd : array of shape (N, 3+C)
        The original point cloud.

    tfm : array of shape (3, 3)
        The transformation matrix.

    Returns
    -------
    new_pcd : array of shape (N, 3+C)
        The transformed point cloud.

    Raises
    ------
    ValueError
        If ``pcd`` or ``tfm`` do not have the expected shape.

    Examples
    --------
    >>> pcd = np.array([[3, -9, 2, 6], [3, 4, -1, 8]])
    >>> tfm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> transform_pcd(pcd, tfm)
    array([[ 9,  3,  2,  6],
           [-4,  3, -1,  8]])

    >>> pcd = np.random.randn(424, 2)  # Invalid shape.
    >>> transform_pcd(pcd, tfm)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (N, 3+C) but got array of shape (424, 2)!
    """
    _check_shape(pcd)

    if not tfm.shape == (3, 3):
        raise ValueError(
                'Expecting array of shape (3, 3) '
                f'but got array of shape {tfm.shape}!'
                )

    coords, feats = split_pcd(pcd)
    new_coords = coords @ tfm.T  # Transpose the matrix.

    return np.hstack((new_coords, feats))


def _center_pcd(pcd):
    r"""
    Center the coordinates of a point cloud by subtracting their centroid.

    .. note::
        The ``features == pcd[:, 3:]`` are not affected.

    Parameters
    ----------
    pcd : array of shape (N, 3+C)

    Returns
    -------
    new_pcd : array of shape (N, 3+C)
        The centered point cloud.

    Raises
    ------
    ValueError
        If ``pcd`` does not have the expected shape.

    Examples
    --------
    >>> pcd = np.array([[2, 1, 3, 9, 6], [-3, 2, 8, 7, 8]])
    >>> new_pcd = _center_pcd(pcd)
    >>> new_pcd.mean(axis=0)
    array([0., 0., 0., 8., 7.])

    >>> pcd = np.random.randn(100, 2)  # Invalid shape.
    >>> new_pcd = _center_pcd(pcd)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (N, 3+C) but got array of shape (100, 2)!
    """
    _check_shape(pcd)

    centroid = pcd.mean(axis=0)
    centroid[3:] = 0  # Center only the coordinates.

    return pcd - centroid


class Center():
    r"""
    Center the coordinates of a point cloud by subtracting their centroid.

    Examples
    --------
    >>> pcd = np.array([[1., 2., 3., 5.], [2., 4., 5., 3.]])
    >>> center = Center()
    >>> center(pcd)
    array([[-0.5, -1. , -1. ,  5. ],
           [ 0.5,  1. ,  1. ,  3. ]])

    >>> center(pcd).mean(axis=0)
    array([0., 0., 0., 4.])
    """
    def __call__(self, pcd):
        return _center_pcd(pcd)  # Checks also for shape.


class Identity():
    r"""
    Leave the point cloud unchanged.

    Examples
    --------
    >>> pcd = np.random.randn(300, 4)
    >>> identity = Identity()
    >>> np.array_equal(identity(pcd), pcd)
    True
    """
    def __call__(self, pcd):
        return pcd


class RandomRotation():
    r"""
    Randomly rotate the coordinates of a point cloud.

    Examples
    --------
    >>> pcd = np.random.randn(25, 4)
    >>> rot = RandomRotation()
    >>> new_pcd = rot(pcd)
    >>> new_pcd.shape
    (25, 4)

    >>> from aidsorb.utils import split_pcd
    >>> coords, feats = split_pcd(pcd)
    >>> new_coords, new_feats = split_pcd(new_pcd)

    >>> np.array_equal(new_coords, coords)  # Coordinates are affected.
    False
    >>> np.array_equal(new_feats, feats)  # Features are not affected.
    True
    """
    def __call__(self, pcd):
        _check_shape(pcd)

        coords, feats = split_pcd(pcd)
        new_coords = R.random().apply(coords)

        return np.hstack((new_coords, feats))


class Jitter():
    r"""
    Jitter the coordinates of a point cloud by adding normal noise.

    Parameters
    ----------
    std: float, default=0.01
        The standard deviation of the normal noise.

    Examples
    --------
    >>> pcd = np.random.randn(100, 5)
    >>> jitter = Jitter()
    >>> new_pcd = jitter(pcd)
    >>> new_pcd.shape
    (100, 5)

    >>> from aidsorb.utils import split_pcd
    >>> coords, feats = split_pcd(pcd)
    >>> new_coords, new_feats = split_pcd(new_pcd)

    >>> np.array_equal(new_coords, coords)  # Coordinates are affected.
    False
    >>> np.array_equal(new_feats, feats)  # Features are not affected.
    True
    """
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, pcd):
        _check_shape(pcd)

        noise = np.random.normal(loc=0, scale=self.std, size=pcd.shape)
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
    >>> pcd = np.random.randn(100, 5)
    >>> erase = RandomErase(n_points=10)
    >>> erase(pcd).shape
    (90, 5)
    """
    def __init__(self, n_points=5):
        self.n_points = n_points

    def __call__(self, pcd):
        _check_shape(pcd)

        # Indices of points to keep.
        keep_size = len(pcd) - self.n_points
        indices = np.random.choice(len(pcd), size=keep_size, replace=False)

        return pcd[indices]
