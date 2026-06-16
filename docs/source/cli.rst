.. highlight:: console

💻 CLI
======

There are two available commands: ``aidsorb`` and ``aidsorb-lit``.

.. tab-set::

    .. tab-item:: aidsorb

        For creating, preparing and visualizing input representations.

        +------------+--------------------------------------------------------------+
        | Subcommand | Short description                                            |
        +============+==============================================================+
        | create     | Create input representations from a directory of structures. |
        +------------+--------------------------------------------------------------+
        | prepare    | Split datat into train, validation and test sets.            |
        +------------+--------------------------------------------------------------+
        | visualize  | Visualize input representations.                             |
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
