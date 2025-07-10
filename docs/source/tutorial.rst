:octicon:`rocket` Tutorial
==========================

.. note::
   **This tutorial covers the most common use cases of AIdsorb**. For more
   advanced usage, you should consult the :doc:`api`.

.. _Introduction:

Introduction
------------

*What is a point cloud?*

   A point cloud is a *set of 3D data points*, i.e. a *set of 3D coordinates
   and (optionally) associated features*. More formally:

   .. math::
      \mathcal{P} = \{\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_N\}
      \quad
      \text{and}
      \quad
      \mathbf{p}_i \in \mathbb{R}^{3+C}

   where :math:`N` is the number of points in the point cloud and :math:`C` is
   the number of (per-point) features.

   In |aidsorb|, a point cloud is represented as a :class:`~numpy.ndarray` or
   :class:`~torch.Tensor` of shape ``(N, 3+C)``:

   .. math::
      \mathcal{P} =
      \begin{bmatrix}
         \mathbf{p}_1 \\
         \mathbf{p}_2 \\
         \vdots \\
         \mathbf{p}_N
      \end{bmatrix}
      =
      \begin{bmatrix}
         x_1 & y_1 & z_1 & f_{1}^1 & \dots & f_1^C \\
         x_2 & y_2 & z_2 & f_{2}^1 & \dots & f_2^C \\
         \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
         x_N & y_N & z_N & f_{N}^1 & \dots & f_N^C \\
      \end{bmatrix}
         

*What is a molecular point cloud?*

   It is a point cloud where coordinates correspond to **atomic positions**,
   and features correspond to **atomic numbers and any additional information**.

   In |aidsorb|, a molecular ``pcd`` is represented as :class:`~numpy.ndarray`
   or :class:`~torch.Tensor` of shape ``(N, 4+C)``, where ``N`` is the number of
   atoms, ``pcd[:, :3]`` are the **atomic coordinates**, ``pcd[:, 3]`` are the
   **atomic numbers** and ``pcd[:, 4:]`` any **additional features**. If ``C ==
   0``, then the only features are the atomic numbers.


.. tip::
   You can visualize a molecular point cloud with:

   .. code-block:: console

      $ aidsorb visualize path/to/structure

Deep learning on molecular point clouds
---------------------------------------

The following components are needed:

* A directory containing files of **molecular structures**.
* A ``.csv`` file containing the **labels of the molecular structures**.
* A ``.yaml`` **configuration file** for orchestrating the DL part.

.. note::
   You are solely responsible for these 3 components.

Data preparation
^^^^^^^^^^^^^^^^

.. rubric:: Create and store the point clouds

Assuming your molecular structures are stored under a directory named
``structures``:

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: console

            $ aidsorb create path/to/structures path/to/pcd_data --features="[en_pauling]"
            $ aidsorb create --config=config.yaml  # Recommended for reproducibility
    
    .. tab-item:: config.yaml

        .. code-block:: yaml

            dirname: 'path/to/structures'
            outname: 'path/to/pcd_data'
            features: ['en_pauling']

    .. tab-item:: Python

        .. code-block:: python

            from aidsorb.utils import pcd_from_dir

            # We add electronegativity as additional feature.
            pcd_from_dir(
                dirname='path/to/structures',
                outname='path/to/pcd_data',
                features=['en_pauling'],
            )

.. rubric:: Split point clouds into train, validation and test sets

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: console

            $ aidsorb prepare path/to/pcd_data --split_ratio="[0.7, 0.1, 0.2]" --seed=1
            $ aidsorb prepare --config=config.yaml  # Recommended for reproducibility

    .. tab-item:: config.yaml

        .. code-block:: yaml

            source: 'path/to/pcd_data'
            split_ratio: [0.7, 0.1, 0.2]
            seed: 1

    .. tab-item:: Python

        .. code-block:: python

            from aidsorb.data import prepare_data

            # Split the data into (train, val, test).
            prepare_data(
                source='path/to/pcd_data',
                split_ratio=(0.7, 0.1, 0.2),
                seed=1,
            )

After creating and splitting the point clouds:

.. code-block:: console

    project_root
    ├── pcd_data
    │   ├── foo.npy
    │   ├── ...
    │   └── bar.npy
    ├── test.json
    ├── train.json
    └── validation.json

.. note::
    * Each ``.npy`` file under ``pcd_data`` corresponds to a point cloud.
    * The ``.json`` files store the point cloud names for training,
      validation and testing.

Train and test
^^^^^^^^^^^^^^

All you need is a ``.yaml`` configuration file and some keystrokes:

.. tab-set::

    .. tab-item:: Train
        
        .. code-block:: console
            
            $ aidsorb-lit fit --config=config.yaml

    .. tab-item:: Test
        
        .. code-block:: console
            
            $ aidsorb-lit test --config=config.yaml --ckpt_path=path/to/ckpt

    .. tab-item:: config.yaml
        
        You can generate and start customizing a configuration file as following::

            $ aidsorb-lit fit --print_config > config.yaml

        Below is a dummy configuration file for multi-output regression using
        PointNet:

        .. literalinclude:: examples/config.yaml
            :language: yaml

    .. tab-item:: labels.csv
        
        .. literalinclude:: examples/labels.csv
            :language: yaml

.. seealso::
    The documentation for the `LightningCLI
    <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html>`_, in case
    you are not familiar with |lightning| and YAML.

.. _Summing up:

Summing up
^^^^^^^^^^

.. code-block:: console

    $ aidsorb create path/to/structures path/to/pcd_data  # Create point clouds
    $ aidsorb prepare path/to/pcd_data  # Split point clouds
    $ aidsorb-lit fit --config=path/to/config.yaml  # Train
    $ aidsorb-lit test --config=path/to/config.yaml --ckpt_path=path/to/ckpt  # Test

Questions
---------

Using point clouds not created with |aidsorb|?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes! The only requirement is to store them under a directory in ``.npy`` format
(see :func:`numpy.save`) and respect the shapes described in
:ref:`Introduction`. Then, you can proceed as described :ref:`earlier <Summing
up>` (omitting the point clouds creation part).

.. _aidsorb_with_pytorch_and_lightning:

Deep learning without the CLI?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Of course! Although you are encouraged to use the :doc:`cli`, you can also use
|aidsorb| with plain |pytorch| or |lightning|.

.. seealso::

    For PyTorch:

    * :class:`aidsorb.data.PCDDataset`
    * :class:`aidsorb.modules`

    For PyTorch Lightning:

    * :class:`aidsorb.datamodules.PCDDataModule`
    * :class:`aidsorb.litmodels.PCDLit`


.. tab-set::

    .. tab-item:: PyTorch

        .. code-block:: python

            from torch.utils.data import DataLoader
            from aidsorb.data import PCDDataset, Collator, get_names
            from aidsorb.modules import PointNet

            # Create the datasets.
            train_set = PCDDataset(pcd_names=get_names('path/to/train.json'), ...)
            val_set = PCDDataset(pcd_names=get_names('path/to/validation.json'), ...)

            # Create the dataloaders.
            train_loader = DataLoader(train_ds, ..., collate_fn=Collator(...))
            val_loader = DataLoader(val_ds, ..., collate_fn=Collator(...))

            # Instantiate the model.
            model = PointNet(...)

            # Your code goes here.
            ...

    .. tab-item:: PyTorch Lightning

        .. code-block:: python

            import lightning as L
            from aidsorb.data import Collator
            from aidsorb.datamodules import PCDDataModule
            from aidsorb.modules import PointNet
            from aidsorb.litmodels import PCDLit

            # Instantiate the datamodule.
            dm = PCDDataModule(
                path_to_X='path/to/pcd_data',
                ...,
                config_dataloaders=dict(collate_fn=Collator(...)),
                )

            # Instantiate the litmodel.
            litmodel = PCDLit(model=PointNet(...), ...)

            # Instantiate the trainer.
            trainer = L.Trainer(...)

            # Your code goes here.
            ...

Predicting directly from the CLI?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, this feature is not available (see :ref:`index:TODO`).
