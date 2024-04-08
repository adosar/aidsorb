r"""
AIdsorb is a Python package for processing molecular point clouds.


A molecular ``pcd`` is an array of shape ``(N, 4+C)`` where ``N`` is the
number of atoms, ``pcd[:, :3]`` are the **atomic coordinates**,
``pcd[:, 3]`` are the **atomic numbers** and ``pcd[:, 3+C]`` any additional
**features**. If ``C==0``, then the only features are the atomic numbers.
"""

__author__ = 'Antonios P. Sarikas'
__copyright__ = 'Copyright (c) 2024 Antonios P. Sarikas'
__license__ = ' GPL-3.0-only'
