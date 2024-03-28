r"""
Add docstring of the module.

The ``pcd`` must have shape of (N, 3+C).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from . utils import center_pcd, transform_pcd


class Centering():
    r"""
    Center a point cloud by subtracting its centroid.

    .. warning::
        The input must be a ``np.array``.

    .. note::
        See also :func:`utils.center_pcd`.

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
        The input must be a ``np.array``.

    .. note::
        See also :func:`utils.transform_pcd`.

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
