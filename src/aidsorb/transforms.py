r"""
Add docstring of the module.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from . utils import center_pcd, transform_pcd


class Centering():
    r"""
    Center a point cloud by subtracting its centroid.

    .. warning::
        The input must be a ``np.array`` of shape ``(N, 4)``.

    .. note::
        See :func:`utils.center_pcd`.

    Parameters
    ----------
    mask_atoms : bool, default=True
        See :func:`utils.center_pcd`.

    Examples
    --------
    >>> x = np.array([[1, 2, 4, 9]])
    >>> center = Centering()
    >>> center(x)
    array([[0., 0., 0., 9.]])

    >>> x = np.array([[1, 2, 4, 9]])
    >>> center = Centering(mask_atoms=False)
    >>> center(x)
    array([[0., 0., 0., 0.]])

    >>> x = np.array([1, 4, 9])
    >>> center = Centering()
    >>> center(x)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (N, 4) but got array of shape (3,)!
    """
    def __init__(self, mask_atoms=True):
        self.mask_atoms = mask_atoms

    def __call__(self, sample):
        new_pcd = center_pcd(sample, mask_atoms=self.mask_atoms)

        return new_pcd


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
        The input must be a ``np.array`` of shape ``(N, 4)``.

    Examples
    --------
    >>> x = np.random.randn(25, 4)
    >>> randomrot = RandomRotation()
    >>> randomrot(x).shape
    (25, 4)

    >>> x = np.random.randn(25, 9)
    >>> randomrot = RandomRotation()
    >>> randomrot(x)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (N, 4) but got array of shape (25, 9)!
    """
    def __call__(self, sample):
        rot = R.random().as_matrix()
        new_pcd = transform_pcd(sample, rot)

        return new_pcd
