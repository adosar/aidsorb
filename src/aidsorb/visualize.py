r"""
Provides helper functions for visualizing molecular point clouds.

.. note::
    You can alternatively visualize a molecular point cloud with ``ase``:

    .. code-block:: python

        from ase.io import read
        from ase.visualize import view

        atoms = read(path/to/file)
        view(atoms)
"""


import numpy as np
import matplotlib.pyplot as plt
from . _check import _check_shape
import plotly.graph_objects as go
from mendeleev.fetch import fetch_table
from . utils import, split_pcd, pcd_from_file


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
    # Subtract 1 to follow the indexing of ``_ptable``.
    atomic_numbers = np.array(atomic_numbers) - 1
    scheme += '_color'

    return _ptable[scheme][atomic_numbers].values


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
    # Subtract 1 to follow the indexing of ``_ptable``.
    atomic_numbers = np.array(atomic_numbers) - 1

    return _ptable['name'][atomic_numbers].values


def draw_pcd_mpl(pcd, scheme='cpk', **kwargs):
    r"""
    Visualize molecular point cloud with Matploblib.

    Each point `pcd[i, :-1]` is colorized and sized based on its atomic
    number `pcd[i, -1]`. For large point clouds, visualization with
    :func:`draw_pcd_plotly` is recommended.

    Parameters
    ----------
    pcd : array of shape (N, 4)
       See :func:`utils.pcd_from_file`.
    scheme : {'jmol', 'cpk'}, default='jmol'
    kwargs
        Valid keyword arguments for `ax.scatter3D`_.

        .. warning::
            Do not pass the arguments ``c`` and ``s``. These are used under the
            hood from :func:`draw_pcd_mpl` to set the size and the color of the
            points based on their atomic number.

    Returns
    -------
    fig : `mpl.figure.Figure`_

    .. _mpl.figure.Figure: https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
    .. _plot.subplots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots
    .. _ax.scatter: https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.scatter.html#mpl_toolkits.mplot3d.axes3d.Axes3D.scatter
    """
    _check_shape(pcd)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    points, atoms = split_pcd(pcd)
    atoms = atoms.ravel()

    colors = get_atom_colors(atoms, scheme=scheme)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=atoms, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return fig


def draw_pcd_plotly(pcd, scheme='cpk', **kwargs):
    r"""
    Visualize molecular point cloud with Plotly.

    Each point `pcd[i, :-1]` is colorized and sized based on its atomic
    number `pcd[i, -1]`.

    Parameters
    ----------
    pcd : array of shape (N, 4)
       See :func:`utils.pcd_from_file`.
    scheme : {'jmol', 'cpk'}, default='jmol'
    kwargs
        Valid keword arguments for `plotly.go.Scatter3D`_.

    Returns
    -------
    fig : `plotly.go.Figure`_

    .. _plotly.go.Figure: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html
    .. _plotly.go.Scatter3D: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scatter3d.html#plotly-graph-objs-scatter3d
    """
    _check_shape(pcd)

    points, atoms = split_pcd(pcd)
    atoms = atoms.ravel()

    colors = get_atom_colors(atoms, scheme=scheme)
    elements = get_elements(atoms)

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker={'size': atoms, 'color': colors},
        hovertext=elements,
        **kwargs
        )])

    return fig


def draw_pcd_from_file(filename, show=True, **kwargs):
    r"""
    Visualize molecular point from a file.

    Parameters
    ----------
    filename : str
        Absolute or relative path to the file.
    show : bool, default=True
        Render the point cloud with ``pio.renderers.default``.
    kwargs
        Valid keyword arguments for :func:`draw_pcd_plotly`.

    Returns
    -------
    render : `plotly.go.Figure`_ if ``show==False`` else ``None``.

    .. _plotly.go.Figure: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html
    """
    _, pcd = pcd_from_file(filename)
    fig = draw_pcd_plotly(pcd, **kwargs)

    return fig.show() if show else fig


# Load the periodic table.
_ptable = fetch_table('elements')
