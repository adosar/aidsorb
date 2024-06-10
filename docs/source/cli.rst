.. highlight:: console

:octicon:`terminal` CLI
=======================

There are two available commands:

.. tab-set::

	.. tab-item:: aidsorb

		For creating, visualizing point clouds and preparing data sets.

		.. code-block::

			$ aidsorb [<subcommand>] --help

	.. tab-item:: aidsorb-lit

		For the machine learning part. Currently, only PointNet is supported.

		.. code-block::

			$ aidsorb-lit [<subcommand>] --help

.. note::
    The ``predict`` subcommand of ``aidsorb-lit`` is currently not available
    (see :ref:`index:TODO`).
