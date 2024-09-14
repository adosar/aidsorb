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
This module provides :class:`torch.nn.Module`'s for building the architectures
in :class:`aidsorb.models`.

Currently, the module provides the basic blocks for building the architecture
from the [PointNet]_ paper.

.. note::
    :class:`PointNetBackbone`, :class:`PointNetClsHead` and
    :class:`PointNetSegHead` have their initial layers *lazy initialized*, so
    you don't need to specify the input dimensionality.

.. warning::
    It is recommended to **use batched inputs in all cases**. For example, even
    if a single ``pcd`` of shape ``(3+C, N)`` is to be passed to
    :class:`PointNetBackbone`, **reshape it to** ``(1, 3+C, N)``. One way to you
    can do it is the following: ``pcd = pcd.unsqueeze(0)``.
"""

import torch
from torch import nn


def conv1d_block(in_channels, out_channels, **kwargs):
    r"""
    Return a 1D convolutional block.

    The block has the following form::

        block = nn.Sequential(
            conv_layer,
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            )

    Parameters
    ----------
    in_channels : int or None
        If ``None``, the ``conv_layer`` is lazy initialized.
    out_channels : int
    **kwargs
        Valid keyword arguments for :class:`torch.nn.Conv1d`.

    Returns
    -------
    block : :class:`torch.nn.Sequential`

    See Also
    --------
    :class:`torch.nn.Conv1d` : For a description of the parameters.

    Examples
    --------
    >>> x = torch.randn(32, 4, 100)  # Shape (B, C_in, N).
    >>> block = conv1d_block(4, 128, kernel_size=1)
    >>> block(x).shape  # Shape (B, C_out, N).
    torch.Size([32, 128, 100])

    >>> # Lazy initialized.
    >>> block = conv1d_block(None, 16, kernel_size=1)
    >>> block(x).shape
    torch.Size([32, 16, 100])
    """
    if in_channels is not None:
        conv_layer = nn.Conv1d(in_channels, out_channels, **kwargs)
    else:
        conv_layer = nn.LazyConv1d(out_channels, **kwargs)

    block = nn.Sequential(
            conv_layer,
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            )

    return block


def dense_block(in_features, out_features, **kwargs):
    r"""
    Return a dense block.

    The block has the following form::

        block = nn.Sequential(
            linear_layer,
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            )

    Parameters
    ----------
    in_features : int or None
        If ``None``, the ``linear_layer`` is lazy initialized.
    out_features : int
    **kwargs
        Valid keyword arguments for :class:`torch.nn.Linear`.

    Returns
    -------
    block : :class:`torch.nn.Sequential`

    See Also
    --------
    :class:`torch.nn.Linear` : For a description of the parameters.

    Examples
    --------
    >>> x = torch.randn(64, 3)  # Shape (B, in_features).
    >>> block = dense_block(3, 10)
    >>> block(x).shape  # Shape (B, out_features).
    torch.Size([64, 10])

    >>> # Lazy initialized.
    >>> block = dense_block(None, 16)
    >>> block(x).shape
    torch.Size([64, 16])
    """
    if in_features is not None:
        linear_layer = nn.Linear(in_features, out_features, **kwargs)
    else:
        linear_layer = nn.LazyLinear(out_features, **kwargs)

    block = nn.Sequential(
            linear_layer,
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            )

    return block


class TNet(nn.Module):
    r"""
    Spatial transformer network (STN) from the [PointNet]_ paper for performing
    the input and feature transform.

    ``T-Net`` takes as input a (possibly embedded) point cloud of shape ``(dim, N)``
    and regresses a ``(dim, dim)`` matrix. Each point in the point cloud has
    shape ``(dim,)``.

    The input must be *batched*, i.e. have shape of ``(B, dim, N)``, where ``B`` is
    the batch size and ``N`` is the number of points in each point cloud.

    Parameters
    ----------
    embed_dim : int
        The embedding dimension.

    Examples
    --------
    >>> tnet = TNet(embed_dim=64)
    >>> x = torch.randn((128, 64, 42))  # Shape (B, embed_dim, N).
    >>> tnet(x).shape
    torch.Size([128, 64, 64])
    """
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        self.conv_blocks = nn.Sequential(
                conv1d_block(embed_dim, 64, kernel_size=1, bias=False),
                conv1d_block(64, 128, kernel_size=1, bias=False),
                conv1d_block(128, 1024, kernel_size=1, bias=False),
                )

        self.dense_blocks = nn.Sequential(
                dense_block(1024, 512, bias=False),
                dense_block(512, 256, bias=False),
                nn.Linear(256, embed_dim * embed_dim),
                )

    def forward(self, x):
        r"""
        Return the regressed matrices.

        Parameters
        ----------
        x : tensor of shape (B, embed_dim, N)

        Returns
        -------
        out : tensor of shape (B, embed_dim, embed_dim)
            The regressed matrices.
        """
        # Input has shape (B, self.embed_dim, N).
        bs = x.shape[0]

        x = self.conv_blocks(x)
        x, _ = torch.max(x, 2, keepdim=False)  # Ignore indices.
        x = self.dense_blocks(x)

        # Initialize the identity matrix.
        identity = torch.eye(self.embed_dim, device=x.device, requires_grad=True).repeat(bs, 1, 1)

        # Output has shape (B, self.embed_dim, self.embed_dim).
        x = x.view(-1, self.embed_dim, self.embed_dim) + identity

        return x


class PointNetBackbone(nn.Module):
    r"""
    Backbone of the :class:`PointNet` model.

    This block is responsible for obtaining the *local and global features*,
    which can then be passed to a task head for predictions. This block also
    returns the *critical indices*.

    The input must be *batched*, i.e. have shape of ``(B, C, N)`` where ``B`` is
    the batch size, ``C`` is the number of input channels  and ``N`` is the
    number of points in each point cloud.

    Parameters
    ----------
    local_feats : bool, default=False
        If ``True``, the returned features are a concatenation of local features
        and global features. Otherwise, the global features are returned.
    n_global_feats : int, default=1024
        The number of global features.

    Examples
    --------
    >>> feat = PointNetBackbone(n_global_feats=2048)
    >>> x = torch.randn((32, 4, 200))
    >>> features, indices = feat(x)
    >>> features.shape
    torch.Size([32, 2048])
    >>> indices.shape
    torch.Size([32, 2048])

    >>> feat = PointNetBackbone(local_feats=True, n_global_feats=1024)
    >>> x = torch.randn((16, 4, 100))
    >>> features, indices = feat(x)
    >>> features.shape
    torch.Size([16, 1088, 100])
    >>> indices.shape
    torch.Size([16, 1024])
    """
    def __init__(self, local_feats=False, n_global_feats=1024):
        super().__init__()

        self.local_feats = local_feats

        # First shared MLP.
        self.shared_mlp_1 = nn.Sequential(
                conv1d_block(None, 64, kernel_size=1, bias=False),
                conv1d_block(64, 64, kernel_size=1, bias=False),
                )

        # Second shared MLP.
        self.shared_mlp_2 = nn.Sequential(
                conv1d_block(64, 64, kernel_size=1, bias=False),
                conv1d_block(64, 128, kernel_size=1, bias=False),
                conv1d_block(128, n_global_feats, kernel_size=1, bias=False),
                )

    def forward(self, x):
        r"""
        Return the *features* and *critical indices*.

        The type of the features is determined by ``local_feats``.

        Parameters
        ----------
        x : tensor of shape (B, C, N)

        Returns
        -------
        out : tuple of length 2
            * ``out[0] == features``
            * ``out[1] == critical_indices``
        """
        n_points = x.shape[2]

        x = self.shared_mlp_1(x)

        if self.local_feats:
            point_feats = x.clone()

        x = self.shared_mlp_2(x)

        # Shape (B, n_global_feats).
        global_feats, critical_indices = torch.max(x, 2, keepdim=False)

        if self.local_feats:
            # Shape (B, n_global_feats + 64, N)
            feats = torch.cat(
            (point_feats, global_feats.unsqueeze(-1).repeat(1, 1, n_points)),
            dim=1
            )

            return feats, critical_indices

        return global_feats, critical_indices


class PointNetClsHead(nn.Module):
    r"""
    Classification head from the [PointNet]_ paper.

    .. note::
        This head can be used for classification or regression.

    Parameters
    ----------
    n_outputs : int, default=1
    dropout_rate : float, default=0

    Examples
    --------
    >>> head = PointNetClsHead(n_outputs=4)
    >>> x = torch.randn(64, 13)
    >>> head(x).shape
    torch.Size([64, 4])
    """
    def __init__(self, n_outputs=1, dropout_rate=0):
        super().__init__()

        self.mlp = nn.Sequential(
                dense_block(None, 512, bias=False),
                dense_block(512, 256, bias=False),
                nn.Dropout(dropout_rate),
                nn.Linear(256, n_outputs),
                )

    def forward(self, x):
        r"""
        Run the forward pass.

        Parameters
        ----------
        x : tensor of shape (B, C)

        Returns
        -------
        out : tensor of shape (B, n_outputs)
        """
        x = self.mlp(x)

        return x


class PointNetSegHead(nn.Module):
    r"""
    Segmentation head from the [PointNet]_ paper.

    .. note::
        This head can be used for segmentation.

    Parameters
    ----------
    n_outputs : int, default=1

    Examples
    --------
    >>> head = PointNetSegHead(n_outputs=2)
    >>> x = torch.randn(32, 1088, 400)
    >>> head(x).shape
    torch.Size([32, 400, 2])
    """
    def __init__(self, n_outputs=1):
        super().__init__()

        self.shared_mlp = nn.Sequential(
                conv1d_block(None, 512, kernel_size=1, bias=False),
                conv1d_block(512, 256, kernel_size=1, bias=False),
                conv1d_block(256, 128, kernel_size=1, bias=False),
                nn.Conv1d(128, n_outputs, kernel_size=1),
                )

    def forward(self, x):
        r"""
        Run the forward pass.

        Parameters
        ----------
        x : tensor of shape (B, C, N).

        Returns
        -------
        out : tensor of shape (B, N, n_outputs)
        """
        x = self.shared_mlp(x)  # Shape (B, n_outputs, N).
        x = x.transpose(2, 1)  # Shape (B, N, n_outputs).

        return x
