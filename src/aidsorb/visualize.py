r"""
Provides helper functions for visualizing molecular point clouds.

.. note::
    You can alternatively visualize a structure with ``ase``:

    .. code-block:: python

        from ase.io import read
        from ase.visualize import view

        atoms = read(path_to_file)
        view(atoms)
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mendeleev.fetch import fetch_table
from . _internal import _check_shape, _ptable
from . utils import split_pcd, pcd_from_file


def get_atom_colors(atomic_numbers, scheme='cpk'):
    r"""
    Convert atomic numbers to colors based on `scheme`.

    Parameters
    ----------
    atomic_numbers : array-like of shape (N,)
    scheme : {'jmol', 'cpk'}, default='jmol'

    Returns
    -------
    colors : array-like of shape (N,)
    """
    atomic_numbers = np.array(atomic_numbers)
    scheme += '_color'

    return _ptable.loc[atomic_numbers, scheme].values


def get_elements(atomic_numbers):
    r"""
    Convert atomic numbers to element names.

    Parameters
    ----------
    atomic_numbers : array-like of shape (N,)

    Returns
    -------
    elements : array-like of shape (N,)
    """
    atomic_numbers = np.array(atomic_numbers)

    return _ptable.loc[atomic_numbers, 'name'].values


def draw_pcd(pcd, scheme='cpk', feature_to_color=None, colorscale=None, **kwargs):
    r"""
    Visualize molecular point cloud with Plotly.

    Each point ``pcd[i]`` is sized based on its atomic number ``pcd[i, 3]``.

    The color of each point is determined by ``feature_to_color``. If ``None``,
    each point is colorized based on its atomic number. Otherwise, it is
    colorized based on its ``pcd[i, feat_idx_label[0]`` value.

    Parameters
    ----------
    pcd : array of shape (N, 3+C)
    scheme : {'jmol', 'cpk'}, default='jmol'
        Takes effect only if ``feature_to_color == None``.
    feature_to_color : tuple of shape (2,), optional
        * ``feature_to_color[0] == idx``, the index of the feature to be colored.
        * ``feature_to_color[1] == label``, the name of the feature for the colorbar.
    colorscale : str, optional
        Takes effect only if ``feature_to_color != None``. See `colorscale`_.
    **kwargs
        Valid keword arguments for `plotly.go.Figure`_.

    Returns
    -------
    fig : `plotly.go.Figure`_

    .. _colorscale: https://plotly.com/python/builtin-colorscales/
    .. _plotly.go.Figure: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html

    Examples
    --------
    >>> pcd = np.random.randint(1, 30, (100, 5))
    >>> fig = draw_pcd(pcd, feature_to_color=(0, 'x coord'), colorscale='viridis')
    """
    _check_shape(pcd)

    points = pcd[:, :3]
    atoms = pcd[:, 3]
    elements = get_elements(atoms)

    if feature_to_color is None:
        colors = get_atom_colors(atoms, scheme=scheme)
        marker = {'size': atoms, 'color': colors}
    else:
        idx, label = feature_to_color
        colors = pcd[:, idx]
        marker = {
                'size': atoms, 'color': colors,
                'colorscale': colorscale,
                'colorbar': {'thickness': 20, 'title': label}
                }

    fig = go.Figure(
            data=[go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=marker,
                hovertext=elements
                )],
            **kwargs
            )

    return fig


def draw_pcd_from_file(filename, show=True, **kwargs):
    r"""
    Visualize molecular point cloud from a file.

    Parameters
    ----------
    filename : str
        Absolute or relative path to the file.
    show : bool, default=True
        Render the point cloud with ``pio.renderers.default``.
    **kwargs
        Valid keyword arguments for :func:`draw_pcd`.

    Returns
    -------
    render : `plotly.go.Figure`_ if ``show == False``, else ``None``.

    .. _plotly.go.Figure: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html

    See Also
    --------
    :func:`draw_pcd`
    """
    _, pcd = pcd_from_file(filename)
    fig = draw_pcd(pcd, **kwargs)

    return fig.show() if show else fig
