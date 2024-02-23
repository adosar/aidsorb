from . utils import _check_shape, split_pcd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mendeleev.fetch import fetch_table


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
    scheme += '_color'

    return _ptable[scheme][atomic_numbers].values


def draw_pcd_mpl(pcd, scheme='cpk', **kwargs):
    r"""
    Visualize a molecular point cloud with Matploblib.

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
    fig : `Figure`_

    .. _Figure: https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
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
    Visualize a molecular point cloud with Plotly.

    Each point `pcd[i, :-1]` is colorized and sized based on its atomic
    number `pcd[i, -1]`.

    Parameters
    ----------
    pcd : array of shape (N, 4)
       See :func:`utils.pcd_from_file`.
    scheme : {'jmol', 'cpk'}, default='jmol'
    kwargs
        Valid keword arguments for `plotly.go.Scatter3D`_

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

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=atoms, color=colors),
        **kwargs
        )])

    return fig


# Load the periodic table.
_ptable = fetch_table('elements')
