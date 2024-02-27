r"""
Docstring of the module.

All models must take input of the form `(B, C, N)` where `B` is the batch size,
`C` the number of input channels (4 for molecular point clouds) and `N` the
number of points in the point cloud.
"""


import torch
import lightning as L
from torch import nn, optim
from torch.nn.functional import max_pool1d


class TNet(nn.Module):
    r"""
    `T-Net` from `PointNet` paper [1]_ for performing the feature transform.

    `T-Net` takes as input a (possibly embedded) point cloud of shape `(dim, N)`
    and regresses to a `(dim, dim)` matrix. Each point in the point cloud has
    shape `(dim,)`.

    The input must be *batched*, i.e. have shape of `(B, dim, N)`, where `B` is
    the batch size.

    Parameters
    ----------
    dim : int, default=64
        The embedding dimension.

    Examples
    --------
    >>> tnet = TNet(dim=64)
    >>> x = torch.randn((128, 64, 42))  # Shape (B, dim, N).
    >>> tnet(x).shape
    torch.Size([128, 64, 64])

    The input must be batched:
    >>> x = torch.randn((64, 42))
    >>> tnet(x)
    Traceback (most recent call last):
        ...
    RuntimeError: running_mean should contain 42 elements not 64

    .. [1] R. Q. Charles, H. Su, M. Kaichun and L. J. Guibas, "PointNet: Deep
    Learning on Point Sets for 3D Classification and Segmentation," 2017 IEEE
    Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI,
    USA, 2017, pp. 77-85, doi: 10.1109/CVPR.2017.16.
    """
    def __init__(self, dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv1d(dim, 64, kernel_size=1),
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
                nn.Linear(256, dim*dim),
                )

        self.dim = dim

    def forward(self, x):
        # Input has shape (B, self.dim, N).
        bs = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=False)[0]  # Get only the values.
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # Initialize the identity matrix.
        identity = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)

        # Output has shape (B, self.dim, self.dim).
        x = x.view(-1, self.dim, self.dim) + identity

        return x


class PointNetFeat(nn.Module):
    r"""
    Backbone for the :class:`PointNet` model.

    This block is responsible for obtaining the **local** and **global
    features**, which can then be passed to a task head for predictions. This
    block also returns the **critical indices** and the **regressed matrices**
    (see :class:`STNkd`).

    The input must be *batched*, i.e. have shape of `(B, C, N)` where `B` is the
    batch size, `C` is the number of input channels (4 for molecular point
    clouds) and `N` is the number of points in the point cloud.

    Parameters
    ----------
    in_channels : int, default=4
        The number of input channels.
    n_global_features : int, default=1024
        The number of `global_features`. These features can be used as input for
        a task head or concatenated with `local_features`.
    local_features : bool, default=False
        If `True`, the returned features are a concatenation of `local_features`
        and `global_features`. Otherwise, only `global_features` are returned.
        
    Notes
    -----
    In this implementation, the input `T-Net` transformation from the original
    paper [1]_ is not applied since is it is not guaranteed to be a rigid
    one.

    Examples
    --------
    >>> feat = PointNetFeat(n_global_features=2048)
    >>> x = torch.randn((32, 4, 239))
    >>> features, indices, A = feat(x)
    >>> features.shape
    torch.Size([32, 2048])
    >>> indices.shape
    torch.Size([32, 2048])
    >>> A.shape
    torch.Size([32, 64, 64])

    >>> feat = PointNetFeat(n_global_features=1024, local_features=True)
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
    def __init__(self, in_channels=4, n_global_features=1024, local_features=False):
        super().__init__()
        self.n_global_features = n_global_features
        self.local_features = local_features

        # T-Net for feature transform.
        self.tnet = TNet(dim=64)

        # The first shared MLP.
        self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                )

        # The second shared MLP.
        self.conv3 = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                )
        self.conv4 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                )
        self.conv5 = nn.Sequential(
                nn.Conv1d(128, self.n_global_features, kernel_size=1),
                nn.BatchNorm1d(self.n_global_features),
                nn.ReLU(),
                )

    def forward(self, x):
        # Input has shape (B, C, N).
        bs = x.shape[0]
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
        x = self.conv5(x)  # Shape (B, self.global_features, N).

        # Shape (B, self.global_features).
        global_features, critical_indices = torch.max(x, 2, keepdim=False)

        if self.local_features:
            # Shape (B, self.global_features + 64, N)
            features = torch.cat(
                    (point_features, global_features.unsqueeze(-1).repeat(1, 1, n_points)),
                    dim=1
                    )
            
            return features, critical_indices, A

        else:
            return global_features, critical_indices, A
