.. highlight:: console

:octicon:`terminal` CLI
=======================

There are two available commands: ``aidsorb`` and ``aidsorb-lit``.

.. tab-set::

    .. tab-item:: aidsorb

        For creating, preparing and visualizing molecular point clouds.

        +------------+--------------------------------------------------------------+
        | Subcommand | Short description                                            |
        +============+==============================================================+
        | create     | Create molecular point clouds from a directory of structures.|
        +------------+--------------------------------------------------------------+
        | prepare    | Split point clouds into train, validation and test sets.     |
        +------------+--------------------------------------------------------------+
        | visualize  | Visualize a point cloud.                                     |
        +------------+--------------------------------------------------------------+

        For more information::

            $ aidsorb [<subcommand>] --help

    .. tab-item:: aidsorb-lit

        For the deep learning part.

        +------------+--------------------------------------------------------------+
        | Subcommand | Short description                                            |
        +============+==============================================================+
        | fit        | Run the full optimization routine.                           |
        +------------+--------------------------------------------------------------+
        | validate   | Evaluate the model on the validation set.                    |
        +------------+--------------------------------------------------------------+
        | test       | Evaluate the model on the test set.                          |
        +------------+--------------------------------------------------------------+
        | predict    | Currently, not available (see :ref:`index:TODO`).            |
        +------------+--------------------------------------------------------------+

        For more information::

            $ aidsorb-lit [<subcommand>] --help
