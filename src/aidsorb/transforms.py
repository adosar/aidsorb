r"""
This module provides helper functions and classes for transforming point clouds.

The ``pcd`` must have shape of ``(N, 3+C)``.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from . utils import split_pcd
from . _internal import _check_shape


def transform_pcd(pcd, tfm):
    r"""
    Transform the coordinates of a point cloud.

    For molecular point clouds, *only rigid transformations are recommended*.

    ..note::
        The ``features == pcd[:, 3:]`` are not affected.

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

    points, features = split_pcd(pcd)
    new_points = points @ tfm.T  # Transpose the matrix.

    return np.hstack((new_points, features))


def center_pcd(pcd):
    r"""
    Center a point cloud.

    The centering is performed by subtracting the centroid (of the points)
    ``centroid == points.mean(axis=0)`` from ``points == pcd[:, :3]``.

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
    >>> new_pcd = center_pcd(pcd)
    >>> new_pcd.mean(axis=0)
    array([0., 0., 0., 8., 7.])
    """
    _check_shape(pcd)

    centroid = pcd.mean(axis=0)
    centroid[3:] = 0  # Center only the coordinates.

    return pcd - centroid


class Centering():
    r"""
    Center the coordinates of a point cloud by subtracting their centroid.

    ..note::
        The ``features == pcd[:, 3:]`` are not affected.

    See Also
    --------
    :func:`center_pcd`

    Examples
    --------
    >>> x = np.array([[1., 2., 3., 4., 5.]])
    >>> center = Centering()
    >>> center(x)
    array([[0., 0., 0., 4., 5.]])

    >>> x = np.array([[1., 4.]])
    >>> center = Centering()
    >>> center(x)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (N, 3+C) but got array of shape (1, 2)!
    """
    def __call__(self, pcd):
        return center_pcd(pcd)


class Identity():
    r"""
    Leave the point cloud unchanged.

    Examples
    --------
    >>> x = np.random.randn(300, 4)
    >>> identity = Identity()
    >>> np.all(identity(x) == x)
    True
    """
    def __call__(self, pcd):
        return pcd


class RandomRotation():
    r"""
    Randomly rotate the coordinates of a point cloud.

    ..note::
        The ``features == pcd[:, 3:]`` are not affected.

    See Also
    --------
    :func:`transform_pcd`

    Examples
    --------
    >>> x = np.random.randn(25, 3)
    >>> randomrot = RandomRotation()
    >>> randomrot(x).shape
    (25, 3)

    >>> x = np.random.randn(25, 2)
    >>> randomrot = RandomRotation()
    >>> randomrot(x)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (N, 3+C) but got array of shape (25, 2)!
    """
    def __call__(self, pcd):
        rot = R.random().as_matrix()
        new_pcd = transform_pcd(pcd, rot)

        return new_pcd


class Jitter():
    r"""
    Jitter the coordinates of a point cloud by adding standard normal noise.

    ..note::
        The ``features == pcd[:, 3:]`` are not affected.

    Parameters
    ----------
    std: float, default=0.001
        The standard deviation of the normal noise.

    Examples
    --------
    >>> pcd = np.random.randn(100, 5)
    >>> jitter = Jitter()
    >>> new_pcd = jitter(pcd)
    >>> np.all(pcd[:, 3:] == new_pcd[:, 3:])  # Features are not affected.
    True
    """
    def __init__(self, std=0.001):
        self.std = std

    def __call__(self, pcd):
        _check_shape(pcd)

        noise = np.random.normal(loc=0, scale=self.std, size=pcd.shape)
        noise[:, 3:] = 0  # Jitter only the coordinates.

        return pcd + noise
