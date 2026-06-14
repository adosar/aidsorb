# This file is part of AIdsorb.
# Copyright (C) 2024-2026 Antonios P. Sarikas

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
Helper functions for visualizing input representations.

.. tip::

    To visualize point clouds or voxels using the CLI:

        .. code-block:: console
            
            $ aidsorb visualize path/to/pcd_or_voxels.npy

    To visualize a structure you can use :mod:`ase`:

        .. code-block:: python

            from ase.io import read
            from ase.visualize import view

            atoms = read('path/to/file')
            view(atoms)
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from plotly.graph_objects import Figure, Scatter3d, Volume

from ._internal import ptable
from .utils import pcd_from_file


def get_atom_colors(atomic_numbers: ArrayLike, scheme: str = 'cpk') -> NDArray:
    r"""
    Convert atomic numbers to colors based on ``scheme``.

    Parameters
    ----------
    atomic_numbers : array-like of shape (N,)
    scheme : {'jmol', 'cpk'}, default='cpk'

    Returns
    -------
    colors : array of shape (N,)
    """
    atomic_numbers = np.array(atomic_numbers)
    scheme += '_color'

    return ptable.loc[atomic_numbers, scheme].to_numpy()


def get_atom_names(atomic_numbers: ArrayLike) -> NDArray:
    r"""
    Convert atomic numbers to element names.

    Parameters
    ----------
    atomic_numbers : array-like of shape (N,)

    Returns
    -------
    elements : array of shape (N,)

    Examples
    --------
    >>> atomic_numbers = np.array([1, 2, 7])
    >>> get_atom_names(atomic_numbers)
    array(['Hydrogen', 'Helium', 'Nitrogen'], dtype=object)
    """
    atomic_numbers = np.array(atomic_numbers)

    return ptable.loc[atomic_numbers, 'name'].to_numpy()


def draw_pcd(
        pcd: ArrayLike,
        molecular: bool = True,
        scheme: str = 'cpk',
        size: float = 2.,
        feature_to_color: tuple[int, str] | None = None,
        colorscale: str | None = None,
        ) -> Figure:
    r"""
    Visualize point cloud with Plotly.

    .. _colorscale: https://plotly.com/python/builtin-colorscales/

    Parameters
    ----------
    pcd : array-like of shape (N, 3+C)
    molecular : bool, default=True
        If :obj:`True`, assume a molecular point cloud. In this case, each atom
        is sized as :math:`r_{\text{vdW}}^4` and colorized based on ``scheme``.
        If :obj:`False`, assume a generic point cloud and size each point based
        on ``size``.
    scheme : {'jmol', 'cpk'}, default='cpk'
        Takes effect only if ``molecular=True`` and ``feature_to_color=None``.
    size : float, default=2.
        Controls the size of points.
    feature_to_color : tuple, optional
        Tuple of the form ``(index, label)``, where ``index`` is the index of
        the feature to be colored and ``label`` is the text label for the
        colorbar.
    colorscale : str, optional
        No effect if ``feature_to_color=None``. For available options, see
        `colorscale`_.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> pcd = np.random.randn(10, 3)
    >>> fig = draw_pcd(pcd, molecular=False, feature_to_color=(0, 'x coord'), colorscale='viridis')
    """
    size = (ptable.loc[pcd[:, 3], 'vdw_radius'] * 0.01)**4 if molecular else size
    hovertext = get_atom_names(pcd[:, 3]) if molecular else None
    color = get_atom_colors(pcd[:, 3], scheme=scheme) if molecular else None
    marker = {'size': size, 'color': color}

    if feature_to_color is not None:
        idx, label = feature_to_color
        marker.update({
            'color': pcd[:, idx],
            'colorscale': colorscale,
            'colorbar': {'thickness': 20, 'title': label},
            })

    fig = Figure(
            data=[Scatter3d(
                x=pcd[:, 0],
                y=pcd[:, 1],
                z=pcd[:, 2],
                mode='markers',
                marker=marker,
                hovertext=hovertext,
                )],
            )

    return fig


def draw_voxels(
        voxels: ArrayLike,
        isomin : float = -1000.,
        isomax : float = 1000.,
        opacity : float = 0.15,
        surface_count: int = 5,
        colorscale : str | None = None,
        label : str | None = 'Energy / K',
        ) -> Figure:
    r"""
    Visualize voxels with Plotly.

    .. todo::
        Add support for multi-channel voxels.

    .. _colorscale: https://plotly.com/python/builtin-colorscales/

    Parameters
    ----------
    voxels : array-like of shape (D, H, W)
    isomin : float, default=-1000
        Minimum value of the displayed range.
    isomax : float, default=1000
        Maximum value of the displayed range.
    opacity : float, default=0.15
        Opacity of the volume rendering.
    surface_count : int, default=5
        Number of isosurfaces to approximate the volume.
    colorscale : str, optional
        For available options, see `colorscale`_.
    label : str, default='Energy / K'
        Text label for the colorbar.

    Returns
    -------
    plotly.graph_objects.Figure

    Notes
    -----
    Default parameter values are chosen to provide a reasonable visualization
    for energy voxels.

    Examples
    --------
    >>> voxels = np.random.randn(3, 3, 3)
    >>> fig = draw_voxels(voxels, colorscale='viridis')
    """
    x, y, z = np.mgrid[
        0:voxels.shape[0],
        0:voxels.shape[1],
        0:voxels.shape[2]
    ]

    fig = Figure(data=Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=np.clip(voxels, isomin, isomax).flatten(),
        isomin=isomin,
        isomax=isomax,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=colorscale,
        colorbar={'title': label},
    ))

    return fig


def draw_from_file(
        filename: str,
        render: bool = True,
        **kwargs
        ) -> Figure | None:
    r"""
    Visualize point cloud or voxels from a file.

    Parameters
    ----------
    filename : str
        Absolute or relative path to a ``.npy``.
    render : bool, default=True
        Whether to render the data with :data:`plotly.io.renderers.default` or
        return the figure object.
    **kwargs
        Valid keyword arguments for :func:`draw_pcd` or :func:`draw_voxels`.

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    data = np.load(filename)
    draw_fn = draw_pcd if data.ndim == 2 else draw_voxels
    fig = draw_fn(data, **kwargs)

    return fig.show() if render else fig
