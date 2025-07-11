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
:class:`torch.nn.Module`'s for point cloud processing.

.. note::
    :class:`PointNetBackbone`, :class:`PointNetClsHead` and
    :class:`PointNetSegHead` have their initial layers *lazy initialized*, so
    you don't need to specify the input dimensionality.

.. warning::
    It is recommended to **use batched inputs in all cases**. For example, even
    if a single ``pcd`` of shape ``(3+C, N)`` is to be passed to
    :class:`PointNetBackbone`, **reshape it to** ``(1, 3+C, N)``. One way to you
    can do it is the following: ``pcd = pcd.unsqueeze(0)``.

.. todo::
    Add more modules for point cloud processing.

References
----------

.. [PointNet] R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
              Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
              Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,
              USA, 2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
"""

from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, Sequential

from ._torch_utils import get_activation


def conv1d_block(
        in_channels: int,
        out_channels: int,
        config_activation: dict[str, str | dict] | None = None,
        **kwargs
        ) -> Sequential:
    r"""
    Return a 1D convolutional block.

    The block has the following form::

        block = nn.Sequential(
            conv_layer,
            nn.BatchNorm1d(out_channels),
            activation_fn
            )

    Parameters
    ----------
    in_channels : int or None
        If :obj:`None`, the ``conv_layer`` is lazy initialized.
    out_channels : int
    config_activation : dict, default=None
        Dictionary for configuring activation function. If :obj:`None`, the
        :class:`~torch.nn.modules.activation.ReLU` activation is used.

        * ``'name'`` activation's class name :class:`str`
        * ``'hparams'`` activation's hyperparameters :class:`dict`
    **kwargs
        Valid keyword arguments for :class:`~torch.nn.Conv1d`.

    Returns
    -------
    block : torch.nn.Sequential

    Examples
    --------
    >>> inp, out = 4, 128
    >>> x = torch.randn(32, 4, 100)  # Shape (B, in_channels, N).
    >>> config_afn = {'name': 'LeakyReLU', 'hparams': {'negative_slope': 0.5}}

    >>> # Default activation function (ReLU).
    >>> block = conv1d_block(inp, out, kernel_size=1)
    >>> block(x).shape
    torch.Size([32, 128, 100])
    >>> block[2]
    ReLU()

    >>> # Custom activation function.
    >>> block = conv1d_block(inp, out, config_afn, kernel_size=1)
    >>> block(x).shape
    torch.Size([32, 128, 100])
    >>> block[2]
    LeakyReLU(negative_slope=0.5)

    >>> # Lazy initialized.
    >>> block = conv1d_block(None, out, kernel_size=1)
    >>> block(x).shape
    torch.Size([32, 128, 100])
    """
    if in_channels is not None:
        conv_layer = nn.Conv1d(in_channels, out_channels, **kwargs)
    else:
        conv_layer = nn.LazyConv1d(out_channels, **kwargs)

    block = nn.Sequential(
            conv_layer,
            nn.BatchNorm1d(out_channels),
            get_activation(config_activation)
            )

    return block


def dense_block(
        in_features: int,
        out_features: int,
        config_activation: dict[str, str | dict] | None = None,
        **kwargs
        ) -> Sequential:
    r"""
    Return a dense block.

    The block has the following form::

        block = nn.Sequential(
            linear_layer,
            nn.BatchNorm1d(out_features),
            activation_fn,
            )

    Parameters
    ----------
    in_features : int or None
        If :obj:`None`, the ``linear_layer`` is lazy initialized.
    out_features : int
    config_activation : dict, default=None
        Dictionary for configuring activation function. If :obj:`None`, the
        :class:`~torch.nn.modules.activation.ReLU` activation is used.

        * ``'name'`` activation's class name :class:`str`
        * ``'hparams'`` activation's hyperparameters :class:`dict`
    **kwargs
        Valid keyword arguments for :class:`~torch.nn.Linear`.

    Returns
    -------
    block : torch.nn.Sequential

    Examples
    --------
    >>> inp, out = 3, 10
    >>> x = torch.randn(64, inp)  # Shape (B, in_features).
    >>> config_afn = {'name': 'SELU', 'hparams': {}}

    >>> # Default activation function (ReLU).
    >>> block = dense_block(inp, out)
    >>> block(x).shape
    torch.Size([64, 10])
    >>> block[2]
    ReLU()

    >>> # Custom activation function.
    >>> block = dense_block(inp, out, config_afn)
    >>> block(x).shape
    torch.Size([64, 10])
    >>> block[2]
    SELU()

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
            get_activation(config_activation)
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
        Embedding dimension.

    Examples
    --------
    >>> tnet = TNet(embed_dim=64)
    >>> x = torch.randn((128, 64, 42))  # Shape (B, embed_dim, N).
    >>> tnet(x).shape
    torch.Size([128, 64, 64])
    """
    def __init__(self, embed_dim: int) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Return the regressed matrices.

        Parameters
        ----------
        x : tensor of shape (B, embed_dim, N)

        Returns
        -------
        out : tensor of shape (B, embed_dim, embed_dim)
            Regressed matrices.
        """
        # Input has shape (B, self.embed_dim, N).
        bs = x.shape[0]

        x = self.conv_blocks(x)
        x, _ = torch.max(x, 2, keepdim=False)  # Ignore indices.
        x = self.dense_blocks(x)

        # Initialize the identity matrix.
        identity = torch.eye(self.embed_dim, device=x.device, requires_grad=x.requires_grad).repeat(bs, 1, 1)

        # Output has shape (B, self.embed_dim, self.embed_dim).
        return x.view(-1, self.embed_dim, self.embed_dim) + identity


class PointNetBackbone(nn.Module):
    r"""
    Backbone of the vanilla version from the [PointNet]_ paper, where
    :class:`TNet`'s have been removed.

    This module extracts features which can then be passed to a task head for
    predictions. This module also returns the *critical indices*.

    The input must be *batched*, i.e. have shape of ``(B, C, N)`` where ``B`` is
    the batch size, ``C`` is the number of input channels  and ``N`` is the
    number of points in each point cloud.

    Parameters
    ----------
    n_global_feats : int, default=1024
        Number of global features.
    local_feats : bool, default=False
        If :obj:`True`, the returned features are a concatenation of local features
        and global features. Otherwise, the global features are returned.

    Examples
    --------
    >>> feat = PointNetBackbone(2048)
    >>> x = torch.randn(32, 4, 200)
    >>> features, indices = feat(x, return_indices=True)
    >>> features.shape
    torch.Size([32, 2048])
    >>> indices.shape
    torch.Size([32, 2048])

    >>> feat = PointNetBackbone(1024, True)
    >>> x = torch.randn(16, 4, 100)
    >>> features, indices = feat(x, return_indices=True)
    >>> features.shape
    torch.Size([16, 1088, 100])
    >>> indices.shape
    torch.Size([16, 1024])

    >>> feat =  PointNetBackbone(512)
    >>> x = torch.randn(8, 3, 50)
    >>> feat(x).shape  # Only features, no critical indices.
    torch.Size([8, 512])
    """
    def __init__(
            self,
            n_global_feats: int = 1024,
            local_feats: bool = False
            ) -> None:
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

    def forward(
            self,
            x: Tensor,
            return_indices: bool = False
            ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Return the *features* and optionally *critical indices*.

        The type of the features is determined by ``local_feats``.

        Parameters
        ----------
        x : tensor of shape (B, C, N)
        return_indices : bool, default=False
            Whether to return critical indices.

        Returns
        -------
        out : tensor or tuple of tensors
            If ``return_indices=False`` the output are the features, otherwise
            tuple of the form ``(features, critical_indices)``.
        """
        n_points = x.shape[2]

        x = self.shared_mlp_1(x)
        point_feats = x
        x = self.shared_mlp_2(x)

        # Shape (B, n_global_feats).
        global_feats, critical_indices = torch.max(x, 2, keepdim=False)
        out = global_feats

        if self.local_feats:
            # Shape (B, n_global_feats + 64, N)
            out = torch.cat((
                point_feats,
                global_feats.unsqueeze(-1).repeat(1, 1, n_points)
                ), dim=1)

        if return_indices:
            return out, critical_indices

        return out


class PointNetClsHead(nn.Module):
    r"""
    Classification head from the [PointNet]_ paper.

    .. tip::
        This head can be used for classification or regression tasks.

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
    def __init__(self, n_outputs: int = 1, dropout_rate: float = 0) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
                dense_block(None, 512, bias=False),
                dense_block(512, 256, bias=False),
                nn.Dropout(dropout_rate),
                nn.Linear(256, n_outputs),
                )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Run the forward pass.

        Parameters
        ----------
        x : tensor of shape (B, C)

        Returns
        -------
        out : tensor of shape (B, n_outputs)
        """
        return self.mlp(x)


class PointNetSegHead(nn.Module):
    r"""
    Segmentation head from the [PointNet]_ paper.

    .. tip::
        This head can be used for segmentation tasks.

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
    def __init__(self, n_outputs: int = 1) -> None:
        super().__init__()

        self.shared_mlp = nn.Sequential(
                conv1d_block(None, 512, kernel_size=1, bias=False),
                conv1d_block(512, 256, kernel_size=1, bias=False),
                conv1d_block(256, 128, kernel_size=1, bias=False),
                nn.Conv1d(128, n_outputs, kernel_size=1),
                )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Run the forward pass.

        Parameters
        ----------
        x : tensor of shape (B, C, N).

        Returns
        -------
        out : tensor of shape (B, N, n_outputs)
        """
        out = self.shared_mlp(x)  # Shape (B, n_outputs, N).

        return out.transpose(2, 1)  # Shape (B, N, n_outputs).


class PointNet(torch.nn.Module):
    r"""
    Vanilla version from the [PointNet]_ paper where :class:`TNet`'s have been
    removed.

    :class:`PointNet` takes as input a point cloud and produces one or more
    outputs.  *The type of the task is determined by* ``head``.

    Currently implemented heads include:

        1. :class:`.PointNetClsHead`: classification and regression
        2. :class:`.PointNetSegHead`: segmentation

    The input must be *batched*, i.e. have shape of ``(B, C, N)`` where ``B`` is
    the batch size, ``C`` is the number of input channels  and ``N`` is the
    number of points in each point cloud.

    .. tip::
        You can define a ``custom_head`` head as a :class:`torch.nn.Module` and
        pass it to ``head``.

        If ``local_features=False``, the input to ``custom_head`` must have the
        same shape as in :meth:`PointNetClsHead.forward`.
        Otherwise, the input to ``custom_head`` must have the same shape as in
        :meth:`PointNetSegHead.forward`.
    
    Parameters
    ----------
    head : torch.nn.Module
    n_global_feats : int, default=1024
    local_feats : bool, default=False

    See Also
    --------
    :class:`modules.PointNetBackbone` :
        For a description of ``local_feats`` and ``n_global_feats``.

    Examples
    --------
    >>> cls_head = PointNetClsHead(n_outputs=2)
    >>> seg_head = PointNetSegHead(n_outputs=10)
    >>> x = torch.randn(32, 4, 300)

    >>> cls_net = PointNet(cls_head, 256)
    >>> cls_net(x).shape
    torch.Size([32, 2])
    >>> cls_net.backbone(x).shape  # Only features.
    torch.Size([32, 256])

    >>> seg_net = PointNet(head=seg_head, n_global_feats=512, local_feats=True)
    >>> seg_net(x).shape
    torch.Size([32, 300, 10])
    >>> seg_net.backbone(x, True)[1].shape  # Features and critical indices.
    torch.Size([32, 512])
    """
    def __init__(
            self,
            head: Module,
            n_global_feats: int = 1024,
            local_feats: bool = False
            ) -> None:
        super().__init__()

        self.backbone = PointNetBackbone(
                n_global_feats=n_global_feats,
                local_feats=local_feats,
                )
        self.head = head

    def forward(self, x: Tensor) -> Any:
        r"""
        Run the forward pass.

        Parameters
        ----------
        x : tensor of shape (B, C, N)

        Returns
        -------
        out : tensor
            Output of ``head``.
        """
        return self.head(self.backbone(x))
