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
This module provides helper functions for creating and handling molecular point
clouds.
"""

import os
from pathlib import Path
import warnings
import numpy as np
from tqdm import tqdm
from ase.io import read
from . _internal import _check_shape, _ptable
warnings.filterwarnings('ignore')


def split_pcd(pcd):
    r"""
    Split a point cloud to coordinates and features.

    .. note::
        The returned arrays are copies.

    Parameters
    ----------
    pcd : array of shape (N, 3+C)

    Returns
    -------
    coords_and_feats : tuple of length 2
        * ``coords_and_feats[0] == coords``, array of shape (N, 3).
        * ``coords_and_feats[1] == feats``, array of shape (N, C).

    Examples
    --------
    >>> pcd = np.random.randn(25, 7)  # Point cloud with 4 features.
    >>> coords, feats = split_pcd(pcd)
    >>> coords.shape
    (25, 3)
    >>> feats.shape
    (25, 4)

    >>> pcd = np.random.randn(15, 3)  # Point cloud with no features.
    >>> coords, feats = split_pcd(pcd)
    >>> coords.shape
    (15, 3)
    >>> feats.shape
    (15, 0)
    """
    _check_shape(pcd)

    return pcd[:, :3].copy(), pcd[:, 3:].copy()


def pcd_from_file(filename, features=None):
    r"""
    Create molecular point cloud from a file.

    The molecular ``pcd`` has shape ``(N, 4+C)`` where ``N`` is the
    number of atoms, ``pcd[:, :3]`` are the **atomic coordinates**,
    ``pcd[:, 3]`` are the **atomic numbers** and ``pcd[:, 4:]`` any
    **additional** ``features``. If ``features=None``, then the only features
    are the atomic numbers.

    .. _periodic table: https://mendeleev.readthedocs.io/en/stable/data.html#data--page-root

    .. todo::
        Add option to drop hydrogen atoms for reducing size of point clouds.

    Parameters
    ----------
    filename : str
        Absolute or relative path to the file.
    features : list of str, optional
        All ``float`` properties from `periodic table`_ are supported.

    Returns
    -------
    name_and_pcd : tuple of length 2
        * ``name_and_pcd[0] == name``.
        * ``name_and_pcd[1] == pcd``.

    Notes
    -----
    * The ``name`` of the molecule is the ``basename`` of ``filename`` with its
      suffix removed.
    * To get a list of the supported chemical file formats see
      :func:`ase.io.read`. Alternatively, you can list them from the command line
      with: ``ase info --formats``.

    Examples
    --------
    >>> # xyz coordinates + atomic number + electronegativity + radius.
    >>> name, pcd = pcd_from_file('path/to/file', features=['en_pauling', 'atomic_radius']) # doctest: +SKIP
    """
    name = Path(filename).stem
    structure = read(filename)

    positions = structure.positions
    atoms = structure.numbers

    if features is not None:
        feats = _ptable.loc[atoms, features].to_numpy()
        pcd = np.hstack((positions, atoms[:, None], feats), dtype='float32')
    else:
        pcd = np.hstack((positions, atoms[:, None]), dtype='float32')

    return  name, pcd


def pcd_from_files(filenames, outname, features=None):
    r"""
    Create molecular point clouds from a list of files and store them.

    The point clouds are stored in ``.npz`` format as key-value pairs. For more
    information on this format, see :func:`numpy.savez`.

    Parameters
    ----------
    filenames : iterable
        An iterable providing the filenames. Absolute or relative paths can be
        used.
    outname : str
        Filename where the data will be stored.
    features: list, optional
        See :func:`pcd_from_file`.

    Notes
    -----
    Molecules that can't be processed are omitted.

    Examples
    --------
    >>> # Create and store the point clouds.
    >>> outname = 'path/to/pcds.npz'
    >>> pcd_from_files(['path/to/mol1.xyz', 'path/to/mol2.cif'], outname=outname)  # doctest: +SKIP
    >>> # Load back and access the point clouds.
    >>> pcds = np.load(outname)  # doctest: +SKIP
    >>> mol1_pcd = pcds['mol1']  # doctest: +SKIP
    """
    fnames = np.fromiter(filenames, dtype=object)

    # Dictionary with names as keys and pcd's as values.
    savez_dict = {}

    for f in tqdm(fnames, desc='\033[32mCreating point clouds\033[0m'):
        try:
            name, pcd = pcd_from_file(f, features=features)
            savez_dict[name] = pcd
        except Exception:
            pass

    # Store the point clouds.
    np.savez_compressed(outname, **savez_dict)


def pcd_from_dir(dirname: str, outname: str, features: list=None):
    r"""
    Create molecular point clouds from a directory and store them.

    The point clouds are stored in ``.npz`` format as key-value pairs. For more
    information on this format, see :func:`numpy.savez`.

    Parameters
    ----------
    dirname : str
        Absolute or relative path to the directory.
    outname : str
        Name of the file where point clouds will be stored.
    features: list, optional
        See :func:`pcd_from_file`.

    Notes
    -----
    Molecules that can't be processed are omitted.

    Examples
    --------
    >>> # Create and store the point clouds.
    >>> outname = 'path/to/pcds.npz'
    >>> pcd_from_dir('path/to/dir', outname=outname)  # doctest: +SKIP
    >>> # Load back and access the point clouds.
    >>> pcds = np.load(outname)  # doctest: +SKIP
    >>> mol1_pcd = pcds['mol1']  # doctest: +SKIP
    """
    fnames = (os.path.join(dirname, f) for f in os.listdir(dirname))

    pcd_from_files(fnames, outname, features)
