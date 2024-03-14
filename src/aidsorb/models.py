r"""
Write the docstring of the module.
"""


import torch
from torch import nn


class TNet(nn.Module):
    r"""
    ``T-Net`` from the ``PointNet`` paper [1]_ for performing the feature
    transform.

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

        self.conv1 = nn.Sequential(
                nn.Conv1d(embed_dim, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU()
                )
        self.conv2 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU()
                )
        self.conv3 = nn.Sequential(
                nn.Conv1d(128, 1024, kernel_size=1),
                nn.BatchNorm1d(1024),
                nn.ReLU()
                )
        self.fc1 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                )
        self.fc3 = nn.Sequential(
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

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=False)[0]  # Get only the values.
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

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
        The embedding dimension of :class:`TNet`.
    n_global_features : int, default=1024
        The number of ``global_features``. These features can be used as input for
        a task head or concatenated with ``local_features``.
    local_features : bool, default=False
        If ``True``, the returned features are a concatenation of
        ``local_features`` and ``global_features``. Otherwise, the
        ``global_features`` are returned.
        
    Notes
    -----
    In this implementation, the input ``T-Net`` transformation from the original
    paper [1]_ is not applied since is it is not guaranteed to be a rigid
    one.

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

    .. [1] R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
    Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,
    USA, 2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
    """
    def __init__(
            self, in_channels=4, embed_dim=64,
            n_global_features=1024, local_features=False
            ):
        super().__init__()

        self.local_features = local_features

        # T-Net for feature transform.
        self.tnet = TNet(embed_dim=embed_dim)

        # The first shared MLP.
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels, embed_dim, kernel_size=1),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                )

        # The second shared MLP.
        self.conv3 = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                )
        self.conv4 = nn.Sequential(
                nn.Conv1d(embed_dim, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                )
        self.conv5 = nn.Sequential(
                nn.Conv1d(128, n_global_features, kernel_size=1),
                nn.BatchNorm1d(n_global_features),
                nn.ReLU(),
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

        # Pass through the first shared MLP.
        x = self.conv1(x)
        x = self.conv2(x)

        # Get regressed matrices from T-Net.
        A = self.tnet(x)  # Shape (B, 64, 64).

        # Perform feature transform.
        x = torch.bmm(x.transpose(2, 1), A).transpose(2, 1)

        # Store point features for later concatenation.
        point_features = x.clone()  # Shape (B, 64, N).

        # Pass through the second shared MLP.
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # Shape (B, self.n_global_features, N).

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
    The classification head from the `PointNet` paper [1]_.

    .. note::
        This head can be used either for classification or regression.

    Parameters
    ----------
    n_inputs : int, default=1024
    n_outputs : int, default=1
    dropout_rate : float, default=0.7

    Examples
    --------
    >>> head = PointNetClsHead(n_inputs=13, n_outputs=4)
    >>> x = torch.randn(64, 13)
    >>> out = head(x)
    >>> out.shape
    torch.Size([64, 4])

    .. [1] R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
    Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,
    USA, 2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
    """
    def __init__(self, n_inputs=1024, n_outputs=1, dropout_rate=0.7):
        super().__init__()

        self.mlp = nn.Sequential(
                nn.Linear(n_inputs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, n_outputs),
                )

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : tensor of shape (B, self.n_inputs)

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

    The final layer is replaced by a global pooling layer followed by a linear
    one.

    Parameters
    ----------
    in_channels : int, default=1088
        The number of input channels.
    n_outputs : int, default=1

    Examples
    --------
    >>> head = PointNetSegHead(n_outputs=2)
    >>> x = torch.randn(32, 1088, 400)
    >>> out, max_idx = head(x)
    >>> out.shape
    torch.Size([32, 2])

    >>> max_idx.shape  # Max indices from the pooling layer.
    torch.Size([32, 128])

    .. [1] R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
    Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,
    USA, 2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
    """
    def __init__(self, in_channels=1088, n_outputs=1):
        super().__init__()

        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels, 512, kernel_size=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                nn.Conv1d(512, 256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                )
        self.conv3 = nn.Sequential(
                nn.Conv1d(256, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                )

        self.linear = nn.Linear(128, n_outputs)

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : tensor of shape (B, self.in_channels, N).

        Returns
        -------
        out : tuple of shape (2,)
            * ``out[0] == predictions`` with shape ``(B, self.n_outputs)``
            * ``out[1] == max_indices`` with shape ``(B, 128)``
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # Shape (B, C, N).
        
        # Perform global pooling and store the max indices.
        x, max_idx = torch.max(x, 2, keepdim=False)

        x = self.linear(x)

        return x, max_idx


class PointNet(nn.Module):
    r"""
    A deep learning architecture for processing point clouds [1]_.

    ``PointNet`` takes as input a point cloud and produces one or more outputs.
    *The type of the task is determined by ``head``*.

    Currently implemented heads include:
    1. :class:`PointNetClsHead`: classification and regression.
    2. :class:`PointNetSegHead`: classification and regression.

    The input must be *batched*, i.e. have shape of ``(B, C, N)`` where ``B`` is
    the batch size, ``C`` is the number of input channels  and ``N`` is the
    number of points in each point cloud.

    You can define a ``custom_head`` head as a :class:``nn.Module`` and
    pass it to ``head``.

    .. warning::
        * If ``local_features == False``, the input to ``custom_head`` must have
        the same shape as in :meth:`PointNetClsHead.forward`.
        * If ``local_features == True``, the input to ``custom_head`` must have
        the same shape as in :meth:`PointNetSegHead.forward`.
    
    Parameters
    ----------
    head : class:`nn.Module` object
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
    >>> (predictions, max_idx), indices, A = pointnet(x)
    >>> predictions.shape
    torch.Size([32, 100])
    >>> max_idx.shape
    torch.Size([32, 128])
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
