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
from . _check import _check_shape
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
        If `pcd` or `M` have not the expected shape.
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

    The centering is performed by removing the centroid of the point cloud.

    Parameters
    ----------
    pcd : array of shape (N, 4)
    mask_atoms : bool, default=True
        Whether to mask atoms when calculating the
        centroid. If ``True``, ``centroid == pcd.mean(axis=0) * mask`` where
        ``mask == array([1, 1, 1, 0])``. Otherwise, ``centroid ==
        pcd.mean(axis=0)``.

    Returns
    -------
    new_pcd : array of shape (N, 4)
        The centered point cloud.
    """
    _check_shape(pcd)

    centroid = pcd.mean(axis=0)
    if mask_atoms:
        centroid *= np.array([1, 1, 1, 0])

    new_pcd = pcd - centroid

    return new_pcd


def pcd_from_file(filename):
    r"""
    Create molecular point cloud from a file.

    The molecular point cloud `pcd` is an array of shape `(N, 4)` where `N` is
    the number of atoms, `pcd[:, :3]` are the **atom positions** and `pcd[:, 3]`
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

    pcd = np.hstack((positions, atoms)).astype('float32')

    return name, pcd


def pcd_from_files(filenames, file, shuffle=False, seed=None):
    r"""
    Create molecular point clouds from a list of files and store them.

    The point clouds are stored in ``.npz`` format as key-value pairs. For more
    information, check `np.savez`_.

    Parameters
    ----------
    filenames : list-like object
        A list of files.
    file : str
        The filename where the data will be stored.
    shuffle : bool, default=False
        If ``True``, the point clouds are shuffled.
    seed : int, optional
        Controls the randomness of the ``rng`` used for shuffling. Takes effect
        only if ``shuffle == True``.

    Notes
    -----
    Molecules that can't be processed are ommited.

    .. _np.savez: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
    """
    filenames = np.fromiter(filenames, dtype=object)

    if shuffle:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(filenames)

    # Dictionary with names as keys and pcd's as values.
    savez_dict = {}

    for f in tqdm(filenames, desc='Creating point clouds'):
        try:
            name, pcd = pcd_from_file(f)
            savez_dict[name] = pcd
        except Exception:
            pass

    # Store the point clouds.
    np.savez(file, **savez_dict)


def pcd_from_dir(dirname, file, shuffle=False, seed=None):
    r"""
    Create molecular point clouds from a directory and store them.

    The point clouds are stored in ``.npz`` format as key-value pairs. For more
    information, check `np.savez`_.

    Parameters
    ----------
    dirname : str
        The name of the directory.
    file : str
        The filename where the data will be stored.
    shuffle : bool, default=False
        If ``True``, the point clouds are shuffled.
    seed : int, optional
        Controls the randomness of the ``rng`` used for shuffling. Takes effect
        only if ``shuffle == True``.

    Notes
    -----
    Molecules that can't be processed are ommited.

    .. _np.savez: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
    """
    filenames = np.fromiter(
            (os.path.join(dirname, f) for f in os.listdir(dirname)),
            dtype=object
            )

    if shuffle:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(filenames)

    # Dictionary with names as keys and pcd's as values.
    savez_dict = {}

    for f in tqdm(filenames, desc='Creating point clouds'):
        try:
            name, pcd = pcd_from_file(f)
            savez_dict[name] = pcd
        except Exception:
            pass

    # Store the point clouds.
    np.savez(file, **savez_dict)


def cli():
    r"""Add docstring"""
    fire.Fire(pcd_from_dir)
