r"""
Provides helper functions for creating molecular point clouds.

The ``pcd`` must have shape ``(N, 4+C)``.
"""

import os
from pathlib import Path
import warnings
import numpy as np
from tqdm import tqdm
from ase.io import read
from . _internal import _check_shape, _SEED
warnings.filterwarnings('ignore')


def split_pcd(pcd):
    r"""
    Split a point cloud to points and features.

    Parameters
    ----------
    pcd : array of shape (N, 3+C)

        .. note::
            At least one feature is required, i.e. ``C >= 1``.

    Returns
    -------
    points_and_features : tuple of shape (2,)
        * ``points_and_features[0] == coords``, array of shape (N, 3).
        * ``points_and_features[1] == atoms``, array of shape (N, C).

    Raises
    ------
    ValueError
        If ``pcd`` does not have the expected shape.

    Examples
    --------
    >>> pcd = np.random.randn(25, 7)  # Point cloud with 4 features.
    >>> points, features = split_pcd(pcd)
    >>> points.shape
    (25, 3)
    >>> features.shape
    (25, 4)
    """
    _check_shape(pcd)

    return pcd[:, :3], pcd[:, 3:]


def pcd_from_file(filename):
    r"""
    Create molecular point cloud from a file.

    The molecular ``pcd`` is an array of shape ``(N, 4)`` where ``N`` is the
    number of atoms, ``pcd[:, :3]`` are the **atomic coordinates**
    and ``pcd[:, 3]`` are the **atomic numbers**.

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
    The ``name`` of the molecule is the ``basename`` of ``filename`` with its
    suffix removed.
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
    filenames : iterable
        An iterable providing the filenames. Absolute or relative paths can be
        used.
    outname : str
        Filename where the data will be stored.
    shuffle : bool, default=False
        If ``True``, the point clouds are shuffled.
    seed : int, default=1
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
    seed : int, default=1
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
