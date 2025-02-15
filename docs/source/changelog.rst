:octicon:`log` Changelog
========================

Version 2.0.0
-------------

.. versionadded:: 2.0.0
    
    * Option for ignoring critical indices in :meth:`.PointNetBackbone.forward`.
    * Support for visualizing ``.npy`` file with :func:`.draw_pcd_from_file` and
      ``aidsorb visualize``.
    * :class:`.RandomSample` and :class:`.RandomFlip` transformations.
    * Support for erasing a fraction of points in :class:`.RandomErase`.
    * Option ``config_activation`` for :func:`.conv1d_block` and
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

    * :class:`.PointNet` moved from :mod:`!models` to :mod:`.modules`.
    * Rename :class:`Jitter` to :class:`.RandomJitter`.
    * :func:`.get_names` now returns :class:`tuple` instead of :class:`list`.
    * Bumped Lightning version to ``>=2.5.0`` (:issue:`29`).
    * :func:`.upsample_pcd` moved from :mod:`.data` to
      :mod:`.transforms` and *now raises error if target size is not
      greater than the original*.
    * :func:`.split_pcd` moved from :mod:`.utils` to :mod:`.transforms` and *no
      longer copies data*.
    * :mod:`.transforms` use :mod:`torch` instead of :mod:`numpy` (:issue:`32`).
    * Remove defaults for :mod:`.transforms` since there is no consensus on
      "good" defaults.
    * Point clouds are stored as plain ``.npy`` files under a directory files
      instead of a single ``.npz`` (:issue:`3`).

.. versionremoved:: 2.0.0

    * :mod:`!models` to simplify codebase and improve project structure.
    * :class:`!PointLit` in favor of :class:`.PCDLit`.
    * :class:`!Identity` since it is equivalent to :class:`torch.nn.Identity`
      (and thus redundant).

Version 1.0.0
-------------

🎂 First release for public usage.
