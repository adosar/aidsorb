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

r"""Helper module for simplifying PyTorch related components."""

from collections.abc import Iterable

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def get_activation(config: dict[str, str | dict] | None = None) -> Module:
    r"""
    Return an instance of activation function.

    Parameters
    ----------
    config : dict, default=None
        Dictionary for configuring activation function. If :obj:`None`, the
        :class:`~torch.nn.modules.activation.ReLU` activation is used.

        * ``'name'`` activation's class name :class:`str`
        * ``'hparams'`` activation's hyperparameters :class:`dict`

    Returns
    -------
    activation

    Examples
    --------
    >>> get_activation()  # ReLU, default activation.
    ReLU()

    >>> config = {'name': 'LeakyReLU', 'hparams': {'negative_slope': 0.05}}
    >>> get_activation(config)
    LeakyReLU(negative_slope=0.05)
    """
    if config is None:
        return torch.nn.ReLU()

    act_cls = getattr(torch.nn.modules.activation, config['name'])
    activation = act_cls(**config['hparams'])

    return activation


def get_optim_cls(name: str) -> type[Optimizer]:
    r"""
    Return optimizer class from name.

    Parameters
    ----------
    name : str
        The class name of the optimizer.

    Returns
    -------
    optim_cls

    Examples
    --------
    >>> optim_cls = get_optim_cls('Adam')
    >>> optim_cls
    <class 'torch.optim.adam.Adam'>
    """
    return getattr(torch.optim, name)


def get_lrs_cls(name: str) -> type[LRScheduler]:
    r"""
    Return scheduler class from name.

    Parameters
    ----------
    name : str
        The class name of the scheduler.

    Returns
    -------
    lrs_cls

    Examples
    --------
    >>> lrs_cls = get_lrs_cls('StepLR')
    >>> lrs_cls
    <class 'torch.optim.lr_scheduler.StepLR'>
    """
    return getattr(torch.optim.lr_scheduler, name)


def get_optimizers(
        params: Iterable,
        config_optim: dict[str, str | dict] | None = None,
        config_lrs: dict[str, str | dict] | None = None,
        ):
    r"""
    Construct optimizer and optionally scheduler.

    Helper funcion for reducing the boilerplate of ``configure_optimizers``.

    Parameters
    ----------
    params : iterable
        Iterable containing the parameters to be optimized.
    config_optim : dict, default=None
        Dictionary for configuring optimizer. If :obj:`None`, the
        :class:`~torch.optim.Adam` optimizer with default hyperparameters is
        used.

        * ``'name'`` optimizer's class name :class:`str`
        * ``'hparams'`` scheduler's hyperparameters :class:`dict`
    config_lrs : dict, optional
        Dictionary for configuring learning rate scheduler.

        * ``'name'`` scheduler's class name :class:`str`
        * ``'hparams'`` scheduler's hyperparameters :class:`dict`
        * ``'config'`` scheduler's config  :class:`dict`

    Returns
    -------
    optimizers : :class:`~torch.optim.Optimizer` or dict
        Single optimizer or dictionary with an ``'optimizer'`` and
        ``'lr_scheduler'`` key.

    Examples
    --------
    >>> # Default behavior (Adam without scheduler).
    >>> parameters = torch.nn.Linear(1, 1).parameters()
    >>> get_optimizers(parameters)
    ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    Adam (
        ...
    )

    >>> # Non-default behavior.
    >>> config_optim = {'name': 'SGD', 'hparams': {'lr': 0.1}}
    >>> config_lrs = {'name': 'StepLR', 'hparams': {'step_size': 2}, 'config': {'interval': 'step'}}

    >>> # Custom optimizer without scheduler.
    >>> parameters = torch.nn.Linear(2, 2).parameters()
    >>> get_optimizers(parameters, config_optim)
    ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    SGD (
        ...
        lr: 0.1
        ...
    )

    >>> # Custom optimizer with scheduler.
    >>> parameters = torch.nn.Linear(2, 2).parameters()
    >>> optimizers = get_optimizers(parameters, config_optim, config_lrs)
    >>> optimizers['optimizer'] # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    SGD (
        ...
        lr: 0.1
        ...
    )
    >>> optimizers['lr_scheduler']['scheduler']  # doctest: +ELLIPSIS
    <torch.optim.lr_scheduler.StepLR object at 0x...>
    >>> optimizers['lr_scheduler']['interval']
    'step'
    """
    if config_optim is None:
        optimizer = torch.optim.Adam(params)
    else:
        optim_cls = get_optim_cls(config_optim['name'])
        optimizer = optim_cls(params, **config_optim['hparams'])

    # Optimization without scheduler.
    if config_lrs is None:
        return optimizer

    # Optimization with scheduler.
    lrs_cls = get_lrs_cls(config_lrs['name'])
    scheduler = lrs_cls(optimizer, **config_lrs['hparams'])

    # Create the config required for lightning.
    lr_scheduler_config = config_lrs['config'].copy()
    lr_scheduler_config.update(scheduler=scheduler)

    return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
