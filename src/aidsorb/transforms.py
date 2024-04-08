r"""
This module provides helper functions and classes for transforming point clouds.

The ``pcd`` must have shape of (N, 4+C).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from . utils import split_pcd
from . _internal import _check_shape


def transform_pcd(pcd, tfm):
    r"""
    Transform a point cloud.

    For molecular point clouds, *only rigid transformations are recommended*.

    Parameters
    ----------
    pcd : array of shape (N, 4+C)
        The original point cloud.

    tfm : array of shape (3, 3) or (T, 3, 3)
        The transformation matrix or matrices.

    Returns
    -------
    new_pcd : array of shape (N, 4+C) or (T, N, 4+C)
        The transformed point cloud or point clouds.

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

    >>> from scipy.spatial.transform import Rotation as R
    >>> pcd = np.random.randn(424, 4)
    >>> tfm = R.random(num=32).as_matrix()
    >>> transform_pcd(pcd, tfm).shape
    (32, 424, 4)

    >>> pcd = np.random.randn(424, 3)  # Invalid shape.
    >>> tfm = np.random.randn(32, 3, 3)  # Valid shape.
    >>> transform_pcd(pcd, tfm)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (N, C) with C >= 4 but got array of shape (424, 3)!

    >>> pcd = np.random.randn(424, 4)  # Valid shape.
    >>> tfm = np.random.randn(32, 4, 3)  # Invalid shape.
    >>> transform_pcd(pcd, tfm)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (3, 3) or (T, 3, 3) but got array of shape (32, 4, 3)!
    """
    _check_shape(pcd)

    if not ((tfm.shape == (3, 3)) or tfm.shape[1:] == (3, 3)):
        raise ValueError(
                'Expecting array of shape (3, 3) or (T, 3, 3) '
                f'but got array of shape {tfm.shape}!'
                )

    points, features = split_pcd(pcd)

    if tfm.shape == (3, 3):
        new_points = points @ tfm.T  # Transpose the matrix.

        return np.hstack((new_points, features))

    size = len(tfm)  # The number of transformations.
    new_points = points @ tfm.transpose([0, 2, 1])  # Transpose the matrices.

    features = features[np.newaxis, :]  # Shape (1, N, C).
    features = np.repeat(features, size, axis=0)  # Shape (size, N, C).

    return np.concatenate((new_points, features), axis=2)


def center_pcd(pcd):
    r"""
    Center a point cloud.

    The centering is performed by subtracting the centroid (of the points)
    ``centroid == points.mean(axis=0)`` from ``points == pcd[:, :3]``.

    .. note::
        The ``features == pcd[:, 3:]`` are not affected.

    Parameters
    ----------
    pcd : array of shape (N, 4+C)

    Returns
    -------
    new_pcd : array of shape (N, 4+C)
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

    points, features = split_pcd(pcd)
    centroid = points.mean(axis=0)

    new_points = points - centroid

    return np.hstack((new_points, features))


class Centering():
    r"""
    Center a point cloud by subtracting its centroid.

    .. warning::
        The input must be a ``np.array`` of shape ``(N, 4+C)``.

    .. note::
        See also :func:`center_pcd`.

    Examples
    --------
    >>> x = np.array([[1., 2., 3., 4., 5.]])
    >>> center = Centering()
    >>> center(x)
    array([[0., 0., 0., 4., 5.]])

    >>> x = np.array([[1., 4., 9.]])
    >>> center = Centering()
    >>> center(x)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (N, C) with C >= 4 but got array of shape (1, 3)!
    """
    def __call__(self, sample):
        return center_pcd(sample)


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
    def __call__(self, sample):
        return sample


class RandomRotation():
    r"""
    Randomly rotate a point cloud.

    .. warning::
        The input must be a ``np.array`` of shape ``(N, 4+C)``.

    .. note::
        See also :func:`transform_pcd`.

    Examples
    --------
    >>> x = np.random.randn(25, 4)
    >>> randomrot = RandomRotation()
    >>> randomrot(x).shape
    (25, 4)

    >>> x = np.random.randn(25, 3)
    >>> randomrot = RandomRotation()
    >>> randomrot(x)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (N, C) with C >= 4 but got array of shape (25, 3)!
    """
    def __call__(self, sample):
        rot = R.random().as_matrix()
        new_pcd = transform_pcd(sample, rot)

        return new_pcd
