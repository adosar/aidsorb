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
This module provides helper functions for creating molecular point clouds.
"""

import os
from pathlib import Path
import warnings
import numpy as np
from tqdm import tqdm
from ase.io import read
from . _internal import _ptable
warnings.filterwarnings('ignore')


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

    return name, pcd


def pcd_from_files(filenames, outname, features=None):
    r"""
    Create molecular point clouds from a list of files and store them.

    The point clouds are stored under ``outname/`` as ``.npy`` files.

    Parameters
    ----------
    filenames : iterable
        An iterable providing the filenames. Absolute or relative paths can be
        used.
    outname : str
        Directory name where the data will be stored.
    features: list, optional
        See :func:`pcd_from_file`.

    Notes
    -----
    Molecules that can't be processed are omitted.

    Examples
    --------
    >>> # Create and store the point clouds.
    >>> outname = 'path/to/pcd_data'
    >>> pcd_from_files(['path/to/foo.xyz', 'path/to/bar.cif'], outname)  # doctest: +SKIP
    >>> # Load back a point cloud.
    >>> pcd = np.load(f'{outname}/foo.npy')  # doctest: +SKIP
    """
    # Create the directory if it doesn't exist.
    os.mkdir(outname)
    print(f'\033[1mSuccessfully created directory: {outname}\033[0m')

    # Create point clouds and store them.
    for f in tqdm(filenames, desc='\033[32;1mCreating point clouds\033[0m'):
        try:
            name, pcd = pcd_from_file(f, features=features)
            pathname = os.path.join(outname, name)
            np.save(pathname, pcd)
        except Exception as e:
            print(e)


def pcd_from_dir(dirname: str, outname: str, features: list=None):
    r"""
    Create molecular point clouds from a directory and store them.

    The point clouds are stored under ``outname/`` as ``.npy`` files.

    Parameters
    ----------
    dirname : str
        Absolute or relative path to the directory.
    outname : str
        Directory name where the data will be stored.
    features: list, optional
        See :func:`pcd_from_file`.

    Notes
    -----
    Molecules that can't be processed are omitted.

    Examples
    --------
    >>> # Create and store the point clouds.
    >>> dirname = 'path/to/structures'
    >>> outname = 'path/to/pcd_data'
    >>> pcd_from_dir(dirname, outname)  # doctest: +SKIP
    """
    fnames = [os.path.join(dirname, f) for f in os.listdir(dirname)]

    pcd_from_files(fnames, outname, features)
