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

.. note::
    Currently, only :class:`PointNet` is implemented, a lightweight version of
    the original architecture where the :class:`~aidsorb.modules.TNet`'s for
    input and feature transforms have been removed.

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

import torch
from . modules import PointNetBackbone


class PointNet(torch.nn.Module):
    r"""
    Vanilla version from the [PointNet]_ paper where :class:`TNet`'s have been
    removed.

    ``PointNet`` takes as input a point cloud and produces one or more outputs.
    *The type of the task is determined by* ``head``.

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
        same shape as in :meth:`.PointNetClsHead.forward`.
        Otherwise, the input to ``custom_head`` must have the same shape as in
        :meth:`.PointNetSegHead.forward`.
    
    Parameters
    ----------
    head : :class:`torch.nn.Module`
    local_feats : bool, default=False
    n_global_feats : int, default=1024

    See Also
    --------
    :class:`~aidsorb.modules.PointNetBackbone` :
        For a description of ``local_feats`` and ``n_global_feats``.

    Examples
    --------
    >>> from aidsorb.modules import PointNetClsHead, PointNetSegHead
    >>> cls_head = PointNetClsHead(n_outputs=2)
    >>> seg_head = PointNetSegHead(n_outputs=10)
    >>> x = torch.randn(32, 4, 300)

    >>> cls_net = PointNet(head=cls_head, n_global_feats=256)
    >>> cls_net(x).shape
    torch.Size([32, 2])
    >>> cls_net.backbone(x)[1].shape  # Critical indices.
    torch.Size([32, 256])

    >>> seg_net = PointNet(head=seg_head, n_global_feats=512, local_feats=True)
    >>> seg_net(x).shape
    torch.Size([32, 300, 10])
    >>> seg_net.backbone(x)[1].shape  # Critical indices.
    torch.Size([32, 512])
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
        out : tensor
            The output of ``head``.
        """
        feats, _ = self.backbone(x)  # Ignore critical indices.
        out = self.head(feats)

        return out
