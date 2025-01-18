:octicon:`log` Changelog
========================

Version 2.0.0
-------------

.. versionadded:: 2.0.0
    
    * Add :class:`~.RandomSample`, a transformation that "clips" a point cloud to
      a fixed size.
    * Support for erasing a fraction of points in :class:`~.RandomErase`.
    * Option ``config_activation`` for :func:`~.conv1d_block` and
      :func:`~.dense_block`.
    * Support to ``.load_from_checkpoint`` without arguments for
      :class:`~.PCDLit` and :class:`~.PCDDataModule`.
    * Support for unlabeled data in :class:`~.PCDDataset`,
      :class:`~.PCDDataModule` and :class:`~.Collator`.
    * Option ``drop_last`` for :class:`~.PCDDataModule`.
    * :class:`~.PCDLit` which supports customization for optimizer and
      scheduler (:issue:`25`).
    * :func:`~.center_pcd` as a functional interface of :class:`~.Center`.

.. versionchanged:: 2.0.0

    * Rename :class:`Jitter` to :class:`~.RandomJitter`.
    * :func:`~.get_names` now returns :class:`tuple` instead of :class:`list`.
    * Bumped Lightning version to ``>=2.5.0`` (:issue:`29`).
    * :func:`~.upsample_pcd` moved from :mod:`~aidsorb.data` to
      :mod:`~aidsorb.transforms` and now raises error if target size is not
      greater than the original.
    * :func:`~.split_pcd` moved from :mod:`~aidsorb.utils` to
      :mod:`~aidsorb.transforms` and *no longer copies data*.
    * :mod:`~.transforms` use :mod:`torch` instead of :mod:`numpy` (:issue:`32`).
    * Remove defaults for :mod:`~aidsorb.transforms` since there is no consensus
      on "good" defaults.
    * Point clouds are stored as plain ``.npy`` files under a directory files
      instead of a single ``.npz`` (:issue:`3`).

.. versionremoved:: 2.0.0

    * ``PointLit`` in favor of :mod:`~.PCDLit`.
    * ``Identity`` from :mod:`~.transforms` since it is equivalent to
      :class:`torch.nn.Identity` (and thus redundant).

Version 1.0.0
-------------

🎂 First release for public usage.
