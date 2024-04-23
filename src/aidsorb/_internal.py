r"""Provides helper functions/data for functions in other modules."""

from mendeleev.fetch import fetch_table


def _check_shape(array):
    r"""
    Check if ``array`` has valid shape to be considered a point cloud.

    Parameters
    ----------
    array

    Raises
    ------
    ValueError
        If ``array.shape != (N, 3+C)``.
    """
    if not ((array.ndim == 2) and (array.shape[1] >= 3)):
        raise ValueError(
                'Expecting array of shape (N, 3+C) '
                f'but got array of shape {array.shape}!'
                )


def _check_shape_vis(array):
    r"""
    Check if ``array`` has valid shape to be considered a molecular point cloud.

    Parameters
    ----------
    array

    Raises
    ------
    ValueError
        If ``array.shape != (N, 4+C)``.
    """
    if not ((array.ndim == 2) and (array.shape[1] >= 4)):
        raise ValueError(
                'Expecting array of shape (N, 4+C) '
                f'but got array of shape {array.shape}!'
                )

# Default value for controlling randomness.
_SEED = 1

# Load the periodic table.
_ptable = fetch_table('elements')
_ptable.set_index('atomic_number', inplace=True)
