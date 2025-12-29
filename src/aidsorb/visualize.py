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
Helper functions for visualizing point clouds.

.. tip::

    To visualize a point cloud from the CLI:

        .. code-block:: console
            
            $ aidsorb visualize path/to/structure_or_pcd  # Structure (.xyz, .cif, etc) or .npy

    You can also visualize a structure with :mod:`ase`:

        .. code-block:: python

            from ase.io import read
            from ase.visualize import view

            atoms = read('path/to/file')
            view(atoms)
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from plotly.graph_objects import Figure, Scatter3d

from ._internal import check_shape, ptable
from .utils import pcd_from_file


def get_atom_colors(atomic_numbers: ArrayLike, scheme: str = 'cpk') -> NDArray:
    r"""
    Convert atomic numbers to colors based on ``scheme``.

    Parameters
    ----------
    atomic_numbers : array-like of shape (N,)
    scheme : {'jmol', 'cpk'}, default='jmol'

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
        pcd: NDArray,
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
    pcd : array of shape (N, 3+C)
    molecular : bool, default=True
        If :obj:`True`, assume a molecular point cloud. In this case, each atom
        is sized as :math:`r_{\text{vdW}}^4` and colorized based on ``scheme``.
        If :obj:`False`, assume a generic point cloud and size each point based
        on ``size``.
    scheme : {'jmol', 'cpk'}, default='jmol'
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
    check_shape(pcd)

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


def draw_pcd_from_file(
        filename: str,
        render: bool = True,
        **kwargs
        ) -> Figure | None:
    r"""
    Visualize point cloud from a file.

    Parameters
    ----------
    filename : str
        Absolute or relative path to a ``.npy`` or structure file.
    render : bool, default=True
        Whether to render the point cloud with
        :data:`plotly.io.renderers.default` or return the figure object.
    **kwargs
        Valid keyword arguments for :func:`draw_pcd`.

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    if filename.endswith('.npy'):
        pcd = np.load(filename)
    else:
        _, pcd = pcd_from_file(filename)

    fig = draw_pcd(pcd, **kwargs)

    return fig.show() if render else fig
