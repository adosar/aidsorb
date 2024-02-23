import numpy as np
from ase.io import read
from pathlib import Path


def _check_shape(array):
    r"""
    Check if `array` has valid shape to be considered a molecular point cloud.

    Parameters
    ----------
    array

    Raises
    ------
    ValueError
        If ``array.shape != (N, 4)``.
    """
    if not ((array.ndim == 2) and (array.shape[1] == 4)):
        raise ValueError(
                'Expecting array of shape (N, 4) '
                f'but got array of shape {array.shape}!'
                )


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

    # Transpose the transformation matrix or matrices.
    if M.shape == (3, 3):
        tfm = M.T
        new_points = points @ tfm

        return np.hstack((new_points, atoms))

    else:
        size = len(M)  # The number of rotations.
        tfm = M.transpose([0, 2, 1])
        new_points = points @ tfm

        atoms = atoms[np.newaxis, :]  # Array of shape (1, N, 1).
        atoms = np.repeat(atoms, size, axis=0)  # Array of shape (size, N, 1).

        print(new_points.shape, atoms.shape)

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

    Supported formats are: ``.cif'' and ``.xyz''.

    Parameters
    ----------
    filename : str
        Absolute or relative path to the file.

    Returns
    -------
    name_and_pcd : tuple of shape (2,)
        `name_and_pcd[0]` is the name of the molecule and `name_and_pcd[1]` is
        the point cloud.

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


def pcd_from_sources():
    r"""
    Create molecular point clouds from multiple sources and store them.
    """
    pass
