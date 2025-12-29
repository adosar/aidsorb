:octicon:`log` Changelog
========================

Version 2.0.0
-------------

.. admonition:: Notable changes

    * :mod:`torch` is now used instead of :mod:`numpy` for controlling the
      randomness in :mod:`.transforms`.
    * New LighthningModule :class:`.PCDLit`, which supports customization for
      optimizer and scheduler.
    * New storing scheme for point clouds, stored as plain ``.npy`` files
      instead of a single ``.npz`` file.

.. versionadded:: 2.0.0
    
    * Support for visualizing a generic point cloud (:issue:`69`).
    * Commands ``aidsorb create`` and ``aidsorb prepare`` now support
      configuration files.
    * Option for ignoring critical indices in :meth:`.PointNetBackbone.forward`.
    * Support for visualizing ``.npy`` file with :func:`.draw_pcd_from_file` and
      ``aidsorb visualize``.
    * :class:`.RandomSample` and :class:`.RandomFlip` transformations.
    * Support for erasing a fraction of points and local patches in
      :class:`.RandomErase`.
    * Support for configuring activation function in :func:`.conv1d_block` and
      :func:`.dense_block`.
    * Support to ``.load_from_checkpoint`` without arguments for
      :class:`.PCDLit` and :class:`.PCDDataModule`.
    * Support for unlabeled data in :class:`.PCDDataset`,
      :class:`.PCDDataModule` and :class:`.Collator`.
    * Option ``drop_last`` for :class:`.PCDDataModule`.
    * :class:`.PCDLit` which supports customization for optimizer and
      scheduler (:issue:`25`).
    * :func:`.center_pcd` as a functional interface of :class:`.Center`.

.. versionchanged:: 2.0.0

    * Renamed :func:`.get_elements` to :func:`.get_atom_names`.
    * Columns in :attr:`.PCDDataset.Y` now follow the order specified
      by the user (:issue:`67`).
    * Renamed :mod:`.litmodels` to :mod:`.litmodules` (:issue:`63`).
    * :class:`.Collator` and :func:`.pad_pcds` now accept keyword-only
      arguments.
    * :class:`.PointNet` moved from :mod:`!models` to :mod:`.modules`.
    * :func:`.get_names` now returns :class:`tuple` instead of :class:`list`.
    * Bumped Lightning version to ``>=2.5.0`` (:issue:`29`).
    * :func:`.upsample_pcd` moved from :mod:`.data` to
      :mod:`.transforms` and *now raises error if target size is not
      greater than the original*.
    * :func:`.split_pcd` moved from :mod:`.utils` to :mod:`.transforms` and *no
      longer copies data*.
    * :mod:`.transforms` now use :mod:`torch` instead of :mod:`numpy` (:issue:`32`).
    * Removed defaults for :mod:`.transforms` since there is no consensus on
      "good" defaults.
    * Point clouds are now stored as plain ``.npy`` files under a directory files
      instead of a single ``.npz`` (:issue:`3`).

.. versionremoved:: 2.0.0

    * :class:`Jitter`, use :class:`.RandomJitter` instead.
    * :mod:`!models` to simplify codebase and improve project structure.
    * :class:`!PointLit`, use :class:`.PCDLit` instead.
    * :class:`!Identity` since it is equivalent to :class:`torch.nn.Identity`
      (and thus redundant).

Version 1.0.0
-------------

🎂 First release for public usage.
