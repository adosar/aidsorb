r"""
Provides helper functions for creating and transforming molecular point clouds.
"""


import os
from pathlib import Path
import warnings
import fire
import numpy as np
from tqdm import tqdm
from ase.io import read
from . _internal import _check_shape, _SEED
warnings.filterwarnings('ignore')


def split_pcd(pcd):
    r"""
    Split a point cloud to points and atoms.

    Parameters
    ----------
    pcd : array of shape (N, 4)

    Returns
    -------
    coords_and_atoms : tuple of shape (2,)
        * ``points_and_atoms[0] == coords``, an array of shape (N, 3).
        * ``points_and_atoms[1] == atoms``, an array of shape (N, 1).

    Examples
    --------
    >>> pcd = np.random.randn(25, 4)
    >>> points, atoms = split_pcd(pcd)
    >>> points.shape
    (25, 3)
    >>> atoms.shape
    (25, 1)
    """
    _check_shape(pcd)

    return pcd[:, :3], pcd[:, 3:]


def transform_pcd(pcd, M):
    r"""
    Transform a point cloud.

    For molecular point clouds, *only rigid transformations are recommended*.

    Parameters
    ----------
    pcd : array of shape (N, 4)
        The original point cloud.
    M : array of shape (3, 3) or (T, 3, 3)
        The transformation matrix or matrices.

    Returns
    -------
    new_pcd : array of shape (N, 4) or (T, N, 4)
        The transformed point cloud or point clouds.

    Raises
    ------
    ValueError
        If ``pcd`` or ``M`` do not have the expected shape.

    Examples
    --------
    >>> pcd = np.array([[3, -9, 2, 6], [3, 4, -1, 8]])
    >>> M = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> transform_pcd(pcd, M)
    array([[ 9,  3,  2,  6],
           [-4,  3, -1,  8]])

    >>> from scipy.spatial.transform import Rotation as R
    >>> M = R.from_euler('z', 90, degrees=True).as_matrix()
    >>> pcd = np.array([[1, 0, 0, 2], [0, 1, 0, 8]])
    >>> transform_pcd(pcd, M).astype('int')
    array([[ 0,  1,  0,  2],
           [-1,  0,  0,  8]])

    >>> pcd = np.random.randn(424, 4)
    >>> M = R.random(num=32).as_matrix()
    >>> transform_pcd(pcd, M).shape
    (32, 424, 4)

    >>> pcd = np.random.randn(424, 3)  # Invalid shape.
    >>> M = np.random.randn(32, 3, 3)  # Valid shape.
    >>> transform_pcd(pcd, M)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (N, 4) but got array of shape (424, 3)!

    >>> pcd = np.random.randn(424, 4)  # Valid shape.
    >>> M = np.random.randn(32, 4, 3)  # Invalid shape.
    >>> transform_pcd(pcd, M)
    Traceback (most recent call last):
        ...
    ValueError: Expecting array of shape (3, 3) or (T, 3, 3) but got array of shape (32, 4, 3)!
    """
    _check_shape(pcd)

    if not ((M.shape == (3, 3)) or M.shape[1:] == (3, 3)):
        raise ValueError(
                'Expecting array of shape (3, 3) or (T, 3, 3) '
                f'but got array of shape {M.shape}!'
                )

    points, atoms = split_pcd(pcd)

    if M.shape == (3, 3):
        tfm = M.T  # Transpose the matrix.

        new_points = points @ tfm

        return np.hstack((new_points, atoms))

    size = len(M)  # The number of rotations.
    tfm = M.transpose([0, 2, 1])  # Transpose the matrices.

    new_points = points @ tfm

    atoms = atoms[np.newaxis, :]  # Shape (1, N, 1).
    atoms = np.repeat(atoms, size, axis=0)  # Shape (size, N, 1).

    return np.concatenate((new_points, atoms), axis=2)


def center_pcd(pcd, mask_atoms=True):
    r"""
    Center a point cloud.

    The centering is performed by subtracting the centroid of the point cloud.

    Parameters
    ----------
    pcd : array of shape (N, 4)
    mask_atoms : bool, default=True
        Whether to mask centroid.
        If ``True``, ``centroid == pcd.mean(axis=0) * mask`` where
        ``mask == array([1, 1, 1, 0])``. Otherwise, ``centroid ==
        pcd.mean(axis=0)``.

    Returns
    -------
    new_pcd : array of shape (N, 4)
        The centered point cloud.

    Examples
    --------
    >>> pcd = np.array([[2, 1, 3, 9], [-3, 2, 8, 7]])
    >>> new_pcd = center_pcd(pcd)
    >>> new_pcd.mean(axis=0)
    array([0., 0., 0., 8.])

    >>> pcd = np.array([[2, 1, 3, 9], [-3, 2, 8, 7]])
    >>> new_pcd = center_pcd(pcd, mask_atoms=False)
    >>> new_pcd.mean(axis=0)
    array([0., 0., 0., 0.])
    """
    _check_shape(pcd)

    centroid = pcd.mean(axis=0)
    if mask_atoms:
        centroid *= np.array([1, 1, 1, 0])

    return pcd - centroid


def pcd_from_file(filename):
    r"""
    Create molecular point cloud from a file.

    The molecular point cloud ``pcd`` is an array of shape ``(N, 4)`` where ``N`` is
    the number of atoms, ``pcd[:, :3]`` are the **atom positions** and ``pcd[:, 3]``
    are the **atomic numbers**.

    .. note::
        To get a list of the supported chemical file formats visit
        `ase.io.read<https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.iread>`_.
        or type ``ase info --formats``. Alternatively, you can list them from
        the command line with ``ase info --formats``.

    Parameters
    ----------
    filename : str
        Absolute or relative path to the file.

    Returns
    -------
    name_and_pcd : tuple of shape (2,)
        * ``name_and_pcd[0] == name``.
        * ``name_and_pcd[1] == pcd``.

    Notes
    -----
    The name of the molecule is the `basename` of `filename` with its suffix
    removed.
    """
    name = Path(filename).stem

    structure = read(filename)
    positions = structure.get_positions()
    atoms = structure.get_atomic_numbers().reshape(len(positions), -1)

    pcd = np.hstack((positions, atoms), dtype='float32')

    return name, pcd


def pcd_from_files(filenames, outname, shuffle=False, seed=_SEED):
    r"""
    Create molecular point clouds from a list of files and store them.

    The point clouds are stored in ``.npz`` format as key-value pairs. For more
    information, check `np.savez`_.

    Parameters
    ----------
    filenames : iterable object
        An iterable object providing the filenames. Absolute or relative
        paths can be used.
    outname : str
        Filename where the data will be stored.
    shuffle : bool, default=False
        If ``True``, the point clouds are shuffled.
    seed : int, optional
        Controls the randomness of the ``rng`` used for shuffling. Takes effect
        only if ``shuffle == True``.

    Notes
    -----
    Molecules that can't be processed are omitted.

    .. _np.savez: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
    """
    fnames = np.fromiter(filenames, dtype=object)

    if shuffle:
        rng = np.random.default_rng(seed=_SEED)
        rng.shuffle(fnames)

    # Dictionary with names as keys and pcd's as values.
    savez_dict = {}

    for f in tqdm(fnames, desc='\033[32mCreating point clouds\033[0m'):
        try:
            name, pcd = pcd_from_file(f)
            savez_dict[name] = pcd
        except Exception:
            pass

    # Store the point clouds.
    np.savez_compressed(outname, **savez_dict)


def pcd_from_dir(dirname, outname, shuffle=False, seed=_SEED):
    r"""
    Create molecular point clouds from a directory and store them.

    The point clouds are stored in ``.npz`` format as key-value pairs. For more
    information, check `np.savez`_.

    Parameters
    ----------
    dirname : str
        Absolute or relative path to the directory.
    outname : str
        Name of the file where point clouds will be stored.
    shuffle : bool, default=False
        If ``True``, the point clouds are shuffled.
    seed : int, optional
        Controls the randomness of the ``rng`` used for shuffling. Takes effect
        only if ``shuffle == True``.

    Notes
    -----
    Molecules that can't be processed are omitted.

    .. _np.savez: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
    """
    fnames = np.fromiter(
            (os.path.join(dirname, f) for f in os.listdir(dirname)),
            dtype=object
            )

    if shuffle:
        rng = np.random.default_rng(seed=_SEED)
        rng.shuffle(fnames)

    # Dictionary with names as keys and pcd's as values.
    savez_dict = {}

    for f in tqdm(fnames, desc='\033[32mCreating point clouds\033[0m'):
        try:
            name, pcd = pcd_from_file(f)
            savez_dict[name] = pcd
        except Exception:
            pass

    # Store the point clouds.
    np.savez_compressed(outname, **savez_dict)


def cli():
    ...
