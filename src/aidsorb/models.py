r"""
Write the docstring of the module.
"""

import torch
from torch import nn


def conv1d_block(in_channels, out_channels, **kwargs):
    r"""
    Return a convolutional block.

    The block has the following form::
        block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, **kwargs),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            )
        
    Parameters
    ----------
    in_channels : int
        See `torch.nn.Conv1d`_.
    out_channels : int
        See `torch.nn.Conv1d`_.
    **kwargs
        Valid keyword arguments for `torch.nn.Conv1d`_.

    Returns
    -------
    block : `torch.nn.Sequential`_

    .. _torch.nn.Conv1d: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    .. _torch.nn.Sequential: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
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
        See `torch.nn.Linear`_.
    out_features : int
        See `torch.nn.Linear`_.
    **kwargs
        Valid keyword arguments for `torch.nn.Linear`_.

    Returns
    -------
    block : `torch.nn.Sequential`_

    .. _torch.nn.Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    .. _torch.nn.Sequential: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    """
    block = nn.Sequential(
            nn.Linear(in_features, out_features, **kwargs),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            )

    return block


class TNet(nn.Module):
    r"""
    ``T-Net`` from the ``PointNet`` paper [1]_ for performing the input and
    feature transform.

    ``T-Net`` takes as input a (possibly embedded) point cloud of shape ``(dim, N)``
    and regresses a ``(dim, dim)`` matrix. Each point in the point cloud has
    shape ``(dim,)``.

    The input must be *batched*, i.e. have shape of ``(B, dim, N)``, where ``B`` is
    the batch size and ``N`` is the number of points in each point cloud.

    Parameters
    ----------
    embed_dim : int, default=64
        The embedding dimension.

    Examples
    --------
    >>> tnet = TNet(embed_dim=64)
    >>> x = torch.randn((128, 64, 42))  # Shape (B, embed_dim, N).
    >>> tnet(x).shape
    torch.Size([128, 64, 64])

    .. [1] R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
    Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,
    USA, 2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
    """
    def __init__(self, embed_dim=64):
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
        x : tensor of shape (B, self.embed_dim, N)
            See :class:`TNet`.

        Returns
        -------
        out : tensor of shape (B, self.embed_dim, self.embed_dim)
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

    This block is responsible for obtaining the *local* and *global*
    **features**, which can then be passed to a task head for predictions. This
    block also returns the **critical indices** and the **regressed matrices**
    (see :class:`TNet`).

    The input must be *batched*, i.e. have shape of ``(B, C, N)`` where ``B`` is
    the batch size, ``C`` is the number of input channels  and ``N`` is the
    number of points in each point cloud.

    Parameters
    ----------
    in_channels : int, default=4
        The number of input channels.
    embed_dim : int, default=64
        The embedding dimension of :class:`TNet` used for feature transform.
    n_global_features : int, default=1024
        The number of ``global_features``. These features can be used as input for
        a task head or concatenated with ``local_features``.
    local_features : bool, default=False
        If ``True``, the returned features are a concatenation of
        ``local_features`` and ``global_features``. Otherwise, the
        ``global_features`` are returned.
        
    Examples
    --------
    >>> feat = PointNetBackbone(n_global_features=2048)
    >>> x = torch.randn((32, 4, 239))
    >>> features, indices, A = feat(x)
    >>> features.shape
    torch.Size([32, 2048])
    >>> indices.shape
    torch.Size([32, 2048])
    >>> A.shape
    torch.Size([32, 64, 64])

    >>> feat = PointNetBackbone(n_global_features=1024, local_features=True)
    >>> x = torch.randn((16, 4, 239))
    >>> features, indices, A = feat(x)
    >>> features.shape
    torch.Size([16, 1088, 239])
    >>> indices.shape
    torch.Size([16, 1024])
    >>> A.shape
    torch.Size([16, 64, 64])
    """
    def __init__(
            self, in_channels=4, embed_dim=64,
            n_global_features=1024, local_features=False
            ):
        super().__init__()

        self.local_features = local_features

        # T-Net for feature transform.
        self.tnet = TNet(embed_dim=embed_dim)

        # First shared MLP.
        self.shared_mlp_1 = nn.Sequential(
                # Change the first block with nn.LazyConv1d when its API stabilizes.
                conv1d_block(in_channels, 64, kernel_size=1, bias=False),
                conv1d_block(64, embed_dim, kernel_size=1, bias=False),
                )

        # Second shared MLP.
        self.shared_mlp_2 = nn.Sequential(
                conv1d_block(embed_dim, 64, kernel_size=1, bias=False),
                conv1d_block(64, 128, kernel_size=1, bias=False),
                conv1d_block(128, n_global_features, kernel_size=1, bias=False),
                )

    def forward(self, x):
        r"""
        Return the features, critical indices and the regressed matrices.

        The type of the features is determined by ``self.local_features``.

        Parameters
        ----------
        x : tensor of shape (B, self.in_channels, N)
            See :class:`PointNetBackbone`.

        Returns
        -------
        out : tuple of shape (3,)
            * ``out[0] == features``
            * ``out[1] == critical_indices``
            * ``out[2] == regressed_matrices``
        """
        # Input has shape (B, C, N).
        n_points = x.shape[2]

        x = self.shared_mlp_1(x)

        A = self.tnet(x)  # Shape (B, 64, 64).
        x = torch.bmm(x.transpose(2, 1), A).transpose(2, 1)
        point_features = x.clone()  # Shape (B, 64, N).

        x = self.shared_mlp_2(x)

        # Shape (B, self.n_global_features).
        global_features, critical_indices = torch.max(x, 2, keepdim=False)

        if self.local_features:
            # Shape (B, self.n_global_features + self.embed_dim, N)
            features = torch.cat(
                    (point_features, global_features.unsqueeze(-1).repeat(1, 1, n_points)),
                    dim=1
                    )

            return features, critical_indices, A

        return global_features, critical_indices, A


class PointNetClsHead(nn.Module):
    r"""
    The classification head from the ``PointNet`` paper [1]_.

    .. note::
        This head can be used either for classification or regression.

    Parameters
    ----------
    in_features : int, default=1024
    n_outputs : int, default=1
    dropout_rate : float, default=0

    Examples
    --------
    >>> head = PointNetClsHead(in_features=13, n_outputs=4)
    >>> x = torch.randn(64, 13)
    >>> out = head(x)
    >>> out.shape
    torch.Size([64, 4])

    .. [1] R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
    Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,
    USA, 2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
    """
    def __init__(self, in_features=1024, n_outputs=1, dropout_rate=0):
        super().__init__()

        self.mlp = nn.Sequential(
                dense_block(in_features, 512, bias=False),
                dense_block(512, 256, bias=False),
                nn.Dropout(dropout_rate),
                nn.Linear(256, n_outputs),
                )
    
    def forward(self, x):
        r"""
        Parameters
        ----------
        x : tensor of shape (B, self.in_features)

        Returns
        -------
        out : tensor of shape (B, self.n_outputs)
        """
        x = self.mlp(x)

        return x


class PointNetSegHead(nn.Module):
    r"""
    Modified segmentation head from the ``PointNet`` paper [1]_.

    .. note::
        This head can be used either for classification or regression.

    The final layer is replaced by a global pooling layer followed by a MLP.

    Parameters
    ----------
    in_channels : int, default=1088
    n_outputs : int, default=1
    dropout_rate : int, default=0

    Examples
    --------
    >>> head = PointNetSegHead(n_outputs=2)
    >>> x = torch.randn(32, 1088, 400)
    >>> out = head(x)
    >>> out.shape
    torch.Size([32, 2])

    .. [1] R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
    Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,
    USA, 2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
    """
    def __init__(self, in_channels=1088, n_outputs=1, dropout_rate=0):
        super().__init__()

        self.shared_mlp = nn.Sequential(
                conv1d_block(in_channels, 512, kernel_size=1, bias=False),
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
        Parameters
        ----------
        x : tensor of shape (B, self.in_channels, N).

        Returns
        -------
        out : tensor of shape (B, self.n_outputs)
        """
        x = self.shared_mlp(x)  # Shape (B, C, N).
        
        # Perform global pooling.
        x, _ = torch.max(x, 2, keepdim=False)  # Ignore indices.

        x = self.mlp(x)

        return x


class PointNet(nn.Module):
    r"""
    A deep learning architecture for processing point clouds [1]_.

    ``PointNet`` takes as input a point cloud and produces one or more outputs.
    *The type of the task is determined by* ``head``.

    Currently implemented heads include:
    1. :class:`PointNetClsHead`: classification and regression.
    2. :class:`PointNetSegHead`: classification and regression.

    The input must be *batched*, i.e. have shape of ``(B, C, N)`` where ``B`` is
    the batch size, ``C`` is the number of input channels  and ``N`` is the
    number of points in each point cloud.

    You can define a ``custom_head`` head as a :class:``torch.nn.Module`` and
    pass it to ``head``.

    .. warning::
        * If ``local_features == False``, the input to ``custom_head`` must have
        the same shape as in :meth:`PointNetClsHead.forward`.
        * If ``local_features == True``, the input to ``custom_head`` must have
        the same shape as in :meth:`PointNetSegHead.forward`.
    
    Parameters
    ----------
    head : class:`nn.Module`
    in_channels : int, default=4
        See :class:`PointNetBackBone`.
    embed_dim : int, default=64
        See :class:`PointNetBackBone`.
    n_global_features : int, default=1024
        See :class:`PointNetBackBone`.
    local_features : bool, default=False
        See :class:`PointNetBackBone`.

    Examples
    --------
    >>> head = PointNetSegHead(in_channels=64+256, n_outputs=100)
    >>> pointnet = PointNet(
    ...     head=head, embed_dim=64,
    ...     n_global_features=256, local_features=True
    ...     )
    >>> x = torch.randn(32, 4, 300)
    >>> predictions, indices, A = pointnet(x)
    >>> predictions.shape
    torch.Size([32, 100])
    >>> indices.shape
    torch.Size([32, 256])
    >>> A.shape
    torch.Size([32, 64, 64])

    .. [1] R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
    Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,
    USA, 2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
    """
    def __init__(
            self, head, in_channels=4,
            embed_dim=64, n_global_features=1024,
            local_features=False,
            ):
        super().__init__()

        self.backbone = PointNetBackbone(
                in_channels=in_channels,
                embed_dim=embed_dim,
                n_global_features=n_global_features,
                local_features=local_features,
                )
        
        self.head = head

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : tensor of shape (B, self.in_channels, N)

        Returns
        -------
        out : tuple of shape (3,)
            * ``out[0] == head_output``
            * ``out[1] == critical_indices``
            * ``out[2] == regressed_matrices``
        """
        features, critical_indices, A = self.backbone(x)
        output = self.head(features)

        return output, critical_indices, A
