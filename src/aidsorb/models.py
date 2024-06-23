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
This module provides deep learning architectures for point cloud processing.

All architectures are implemented as :class:`torch.nn.Module`'s. Currently, the
module provides the basic blocks for  for building the architecture
introduced in the [PointNet]_ paper. It also provides :class:`PointNet`, a
lightweight version of the original architecture, where the :class:`TNet`'s for
input and feature transforms have been removed.

.. note::
    :class:`PointNetBackbone`, :class:`PointNetClsHead` and
    :class:`PointNetSegHead` have their initial layers *lazy initialized*, so
    you don't need to specify the input dimensionality.

.. warning::
    It is recommended to **use batched inputs in all cases**. For example, even
    if a single ``pcd`` of shape ``(3+C, N)`` is to be processed with
    :class:`PointNet`, **reshape it to** ``(1, 3+C, N)``. One way to you can do it
    is the following: ``pcd = pcd.unsqueeze(0)``.

.. todo::
    Add more architectures for point cloud processing.

References
----------

.. [PointNet] R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
                    Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
                    Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,
                    USA, 2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
"""

import warnings
import torch
from torch import nn
warnings.filterwarnings('ignore')


def conv1d_block(in_channels, out_channels, **kwargs):
    r"""
    Return a 1D convolutional block.

    The block has the following form::

        block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, **kwargs),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            )

    Parameters
    ----------
    in_channels : int
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
    >>> block = conv1d_block(4, 128, kernel_size=1)
    >>> x = torch.randn(32, 4, 100)  # Shape (B, C_in, N).
    >>> # Shape (B, C_out, N).
    >>> block(x).shape
    torch.Size([32, 128, 100])
    """
    block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, **kwargs),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            )

    return block


def dense_block(in_features, out_features, **kwargs):
    r"""
    Return a dense block.

    The block has the following form::

        block = nn.Sequential(
            nn.Linear(in_features, out_features, **kwargs),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            )

    Parameters
    ----------
    in_features : int
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
    >>> block = dense_block(3, 10)
    >>> x = torch.randn(64, 3)
    >>> block(x).shape
    torch.Size([64, 10])
    """
    block = nn.Sequential(
            nn.Linear(in_features, out_features, **kwargs),
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
        identity = torch.eye(self.embed_dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            identity = identity.to(device='cuda')

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
    >>> x = torch.randn((32, 4, 239))
    >>> features, indices = feat(x)
    >>> features.shape
    torch.Size([32, 2048])
    >>> indices.shape
    torch.Size([32, 2048])

    >>> feat = PointNetBackbone(n_global_feats=1024, local_feats=True)
    >>> x = torch.randn((16, 4, 239))
    >>> features, indices = feat(x)
    >>> features.shape
    torch.Size([16, 1088, 239])
    >>> indices.shape
    torch.Size([16, 1024])
    """
    def __init__(self, local_feats=False, n_global_feats=1024):
        super().__init__()

        self.local_feats = local_feats

        # First shared MLP.
        self.shared_mlp_1 = nn.Sequential(
                nn.LazyConv1d(64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                conv1d_block(64, 64, kernel_size=1, bias=False),
                )

        # Second shared MLP.
        self.shared_mlp_2 = nn.Sequential(
                nn.LazyConv1d(64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(),
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
        out : tuple of shape (2,)
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
    The classification head from the [PointNet]_ paper.

    .. note::
        This head can be used either for classification or regression.

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
                nn.LazyLinear(512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(),
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
    Modified segmentation head from the [PointNet]_ paper.

    The final layer is replaced by a global pooling layer followed by a MLP.

    .. note::
        This head can be used either for classification or regression.

    Parameters
    ----------
    n_outputs : int, default=1
    dropout_rate : int, default=0

    Examples
    --------
    >>> head = PointNetSegHead(n_outputs=2)
    >>> x = torch.randn(32, 1088, 400)
    >>> head(x).shape
    torch.Size([32, 2])
    """
    def __init__(self, n_outputs=1, dropout_rate=0):
        super().__init__()

        lazy_conv1d_block = nn.Sequential(
                nn.LazyConv1d(512, kernel_size=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                )

        self.shared_mlp = nn.Sequential(
                lazy_conv1d_block,
                conv1d_block(512, 256, kernel_size=1, bias=False),
                conv1d_block(256, 128, kernel_size=1, bias=False),
                )

        self.mlp = nn.Sequential(
                dense_block(128, 64, bias=False),
                dense_block(64, 32, bias=False),
                nn.Dropout(dropout_rate),
                nn.Linear(32, n_outputs),
                )

    def forward(self, x):
        r"""
        Run the forward pass.

        Parameters
        ----------
        x : tensor of shape (B, C, N).

        Returns
        -------
        out : tensor of shape (B, n_outputs)
        """
        x = self.shared_mlp(x)  # Shape (B, C, N).

        # Perform global pooling.
        x, _ = torch.max(x, 2, keepdim=False)  # Ignore indices.

        x = self.mlp(x)

        return x


class PointNet(nn.Module):
    r"""
    Vanilla version from the [PointNet]_ paper where :class:`TNet`'s have been
    removed.

    ``PointNet`` takes as input a point cloud and produces one or more outputs.
    *The type of the task is determined by* ``head``.

    Currently implemented heads include:

        1. :class:`PointNetClsHead`: classification and regression.
        2. :class:`PointNetSegHead`: classification and regression.

    The input must be *batched*, i.e. have shape of ``(B, C, N)`` where ``B`` is
    the batch size, ``C`` is the number of input channels  and ``N`` is the
    number of points in each point cloud.

    .. tip::
        You can define a ``custom_head`` head as a :class:`torch.nn.Module` and
        pass it to ``head``.

        If ``local_features=False``, the input to ``custom_head`` must have the
        same shape as in :meth:`PointNetClsHead.forward`. Otherwise, the input
        to ``custom_head`` must have the same shape as in
        :meth:`PointNetSegHead.forward`.
    
    Parameters
    ----------
    head : class:`nn.Module`
    local_feats: bool, default=False
    n_global_feats: int, default=1024

    See Also
    --------
    :class:`PointNetBackBone` :
        For a description of ``local_feats`` and ``n_global_feats``.

    Examples
    --------
    >>> head = PointNetSegHead(n_outputs=100)
    >>> pointnet = PointNet(head=head, local_feats=True, n_global_feats=256)
    >>> x = torch.randn(32, 4, 300)
    >>> out, indices = pointnet(x)
    >>> out.shape
    torch.Size([32, 100])
    >>> indices.shape
    torch.Size([32, 256])
    """
    def __init__(self, head, local_feats=False, n_global_feats=1024):
        super().__init__()
        self.backbone = PointNetBackbone(local_feats, n_global_feats)
        self.head = head

    def forward(self, x):
        r"""
        Run the forward pass.

        Parameters
        ----------
        x : tensor of shape (B, C, N)

        Returns
        -------
        out : tuple of shape (2,)
            * ``out[0] == head_output``
            * ``out[1] == critical_indices``
        """
        feats, indices = self.backbone(x)
        out = self.head(feats)

        return out, indices
