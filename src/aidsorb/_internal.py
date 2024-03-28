r"""Provides functions used for checking parameters in other modules."""


def _check_shape(array):
    r"""
    Check if ``array`` has valid shape to be considered a point cloud.

    Parameters
    ----------
    array

    Raises
    ------
    ValueError
        If ``array.shape != (N, C)`` with ``C >= 4``.
    """
    if not ((array.ndim == 2) and (array.shape[1] >= 4)):
        raise ValueError(
                'Expecting array of shape (N, C) with C >= 4 '
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
        If ``array.shape != (N, 4)``.
    """
    if not ((array.ndim == 2) and (array.shape[1] == 4)):
        raise ValueError(
                'Expecting array of shape (N, 4) '
                f'but got array of shape {array.shape}!'
                )

_SEED = 1  # Default value for controlling randomness.
