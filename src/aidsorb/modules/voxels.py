# This file is part of AIdsorb.
# Copyright (C) 2026 Antonios P. Sarikas

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
:class:`torch.nn.Module`'s for voxels processing.

References
----------

.. [IntelliPore] A. P. Sarikas, K. Gkagkas, and G. E. Froudakis, “IntelliPore: A
                 Foundation Model for Gas Adsorption in Porous Materials"
.. [RetNeXt] A. P. Sarikas, K. Gkagkas, and G. E. Froudakis, “RetNeXt: A
             Pretrained Model for Transfer Learning Across the MOF Adsorption
             Space,” Journal of Chemical Information and Modeling, vol. 66, no.
             4, pp. 2110–2116, Feb. 2026, doi: 10.1021/acs.jcim.5c02698.
"""

from collections import OrderedDict

import torch
from torch import nn, Tensor

from .._torch_utils import get_activation


def conv3d_block(
        in_channels: int,
        out_channels: int,
        config_activation: dict[str, str | dict] | None = None,
        **kwargs
        ):
    r"""
    Return a 3D convolutional block.

    The block has the following form::

        block = nn.Sequential(
            conv_layer,
            nn.BatchNorm3d(out_channels),
            activation_fn
            )

    Parameters
    ----------
    in_channels : int
    out_channels : int
    config_activation : dict, default=None
        Dictionary for configuring activation function. If :obj:`None`, the
        :class:`~torch.nn.modules.activation.ReLU` activation is used.

        * ``'name'`` activation's class name :class:`str`
        * ``'hparams'`` activation's hyperparameters :class:`dict`
    **kwargs
        Valid keyword arguments for :class:`~torch.nn.Conv3d`.

    Returns
    -------
    block : torch.nn.Sequential

    Examples
    --------
    >>> inp, out = 4, 8
    >>> x = torch.randn(2, inp, 3, 3, 3)  # Shape (B, in_channels, H, W, D).
    >>> config_afn = {'name': 'LeakyReLU', 'hparams': {'negative_slope': 0.9}}

    >>> # Default activation function (ReLU).
    >>> block = conv3d_block(inp, out, kernel_size=3)
    >>> block(x).shape
    torch.Size([2, 8, 1, 1, 1])
    >>> block[2]
    ReLU()

    >>> # Custom activation function.
    >>> block = conv3d_block(inp, out, config_afn, kernel_size=3)
    >>> block(x).shape
    torch.Size([2, 8, 1, 1, 1])
    >>> block[2]
    LeakyReLU(negative_slope=0.9)
    """
    return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, **kwargs),
            nn.BatchNorm3d(out_channels),
            get_activation(config_activation),
            )


class RetNeXt(nn.Module):
    r"""
    Architecture from the [RetNeXt]_ paper.

    .. note::
        ``pretrained=True`` is only compatible with ``in_channels=1``, since the
        pretrained backbone was trained on single-channel images.

    Parameters
    ----------
    in_channels : int, default=1
    n_outputs : int or None, default=1
    pretrained : bool, default=False
        Whether to use pretrained weights for the backbone.

    Examples
    --------
    >>> model = RetNeXt(3, 100)
    >>> x = torch.randn(8, 3, 32, 32, 32)
    >>> model(x).shape  # Outputs
    torch.Size([8, 100])
    >>> model.backbone(x).shape  # Embeddings
    torch.Size([8, 128])

    >>> # Works with different grid sizes (adaptive pooling).
    >>> x = torch.randn(16, 3, 25, 25, 25)
    >>> model(x).shape
    torch.Size([16, 100])

    >>> # Pretrained weights for the backbone.
    >>> model = RetNeXt(n_outputs=None, pretrained=True)  # doctest: +ELLIPSIS
    >>> x = torch.randn(2, 1, 32, 32, 32)
    >>> model(x).shape  # Embeddings
    torch.Size([2, 128])
    """
    def __init__(
            self,
            in_channels: int = 1,
            n_outputs: int | None = 1,
            *,
            pretrained: bool = False,
            ):
        super().__init__()

        self.backbone = nn.Sequential(
                nn.BatchNorm3d(in_channels, affine=False, momentum=None),
                conv3d_block(in_channels, 32, kernel_size=3, bias=False, padding='same'),
                conv3d_block(32, 32, kernel_size=3, bias=False, padding='same'),
                nn.MaxPool3d(kernel_size=2), # 1st pooling layer
                conv3d_block(32, 64, kernel_size=3, bias=False, padding='same'),
                conv3d_block(64, 64, kernel_size=3, bias=False, padding='same'),
                nn.MaxPool3d(kernel_size=2),  # 2nd pooling layer
                conv3d_block(64, 128, kernel_size=3, bias=False),
                conv3d_block(128, 128, kernel_size=3, bias=False),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                )

        if pretrained:
            self.backbone.load_state_dict(self.get_pretrained_weights())

        self.head = torch.nn.Identity() if n_outputs is None else nn.Linear(128, n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Run the forward pass.

        Parameters
        ----------
        x : tensor of shape (B, C, H, W, D)

        Returns
        -------
        out : tensor
            If ``n_outputs=None`` return the embeddings of shape ``(B, 128)``,
            else the model outputs of shape ``(B, n_outputs)``.
        """
        return self.head(self.backbone(x))

    def get_pretrained_weights(self) -> OrderedDict:
        r"""
        Return the state dict of the pretrained backbone.
        """
        url = 'https://raw.githubusercontent.com/adosar/retnext-paper/master/pretrained_weights/retnext_cubic_boltzmann_final_all.pt'
        return torch.hub.load_state_dict_from_url(url)


class IntelliPore(nn.Module):
    r"""
    Architecture from the [IntelliPore]_ paper.

    .. note::
        ``pretrained=True`` is only compatible with ``in_channels=1``, since the
        pretrained backbone was trained on single-channel images.

    Parameters
    ----------
    in_channels : int, default=1
    n_outputs : int or None, default=1
    pretrained : bool, default=False
        Whether to use pretrained weights for the backbone.

    Examples
    --------
    >>> model = IntelliPore(3, 100)
    >>> x = torch.randn(4, 3, 32, 32, 32)
    >>> model(x).shape  # Outputs
    torch.Size([4, 100])
    >>> model.backbone(x).shape  # Embeddings
    torch.Size([4, 128])

    >>> # Works with different grid sizes (adaptive pooling).
    >>> x = torch.randn(8, 3, 24, 24, 24)
    >>> model(x).shape
    torch.Size([8, 100])

    >>> # Pretrained weights for the backbone.
    >>> model = IntelliPore(n_outputs=None, pretrained=True)  # doctest: +ELLIPSIS
    >>> x = torch.randn(4, 1, 32, 32, 32)
    >>> model(x).shape  # Embeddings
    torch.Size([4, 128])
    """
    def __init__(
            self,
            in_channels: int = 1,
            n_outputs: int | None = 1,
            *,
            pretrained: bool = False,
            ):
        super().__init__()

        self.backbone = nn.Sequential(
                conv3d_block(in_channels, 32, kernel_size=3, padding='same'),
                conv3d_block(32, 32, kernel_size=3, padding='same'),
                nn.MaxPool3d(kernel_size=2), # 1st pooling layer
                conv3d_block(32, 64, kernel_size=3, padding='same'),
                conv3d_block(64, 64, kernel_size=3, padding='same'),
                nn.MaxPool3d(kernel_size=2),  # 2nd pooling layer
                conv3d_block(64, 128, kernel_size=3),
                conv3d_block(128, 128, kernel_size=3),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten()
                )

        if pretrained:
            self.backbone.load_state_dict(self.get_pretrained_weights())

        self.head = torch.nn.Identity() if n_outputs is None else nn.Linear(128, n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Run the forward pass.

        Parameters
        ----------
        x : tensor of shape (B, C, H, W, D)

        Returns
        -------
        out : tensor
            If ``n_outputs=None`` return the embeddings of shape ``(B, 128)``,
            else the model outputs of shape ``(B, n_outputs)``.
        """
        return self.head(self.backbone(x))

    def get_pretrained_weights(self) -> OrderedDict:
        r"""
        Return the state dict of the pretrained backbone.
        """
        #url = 'https://raw.githubusercontent.com/adosar/intellipore/master/pretrained_weights/intellipore_backbone_pretrained.pt'
        # Temporary needs to be changed
        url = 'https://raw.githubusercontent.com/adosar/trial/master/pretrained_weights/intellipore_backbone_pretrained.pt'
        return torch.hub.load_state_dict_from_url(url)
