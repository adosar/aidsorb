.. AIdsorb documentation master file, created by
   sphinx-quickstart on Thu May 30 15:33:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/aidsorb_logo_light.svg
	:width: 400
	:align: center

|

About
=====

**AIdsorb** is a :fa:`python; fa-fade` Python package for **Machine Learning on
molecular point clouds**.

* What is a point cloud?
* Why it useful in chemistry?
* Why it useful in porous materials (voxels etc.)?
* Do not be fooled by the name.

Objective
---------

Add a small synopsis of the aim of AIdsorb.

.. grid:: 3

	.. grid-item::

		.. button-link:: https://github.com/adosar/aidsorb
			:color: primary
			:align: center

			:octicon:`mark-github` Source code

	.. grid-item::

		.. button-ref:: installation
			:ref-type: doc
			:color: primary
			:align: center

			🚀 Get started

	.. grid-item::

		.. button-link:: https://example.com
			:color: primary
			:align: center

			:octicon:`telescope` AIdsorb paper

Citing AIdsorb
--------------

If you use AIdosrb in your research, please consider citing the following work::

	Currently, not available.

TODO
----

.. card-carousel:: 2

    .. card:: Extend the :doc:`cli`
        :text-align: center

        Allow users to make predictions from the command line.

    .. card:: Add more architectures
        :text-align: center

        This might require the usage of
        :bdg-link-primary:`PyTorch Geometric
        <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`.

    .. card:: Extend featurization
        :text-align: center

        Add more featurization options.

License
-------

AIdsorb is released under the :bdg-link-primary:`GNU General Public License v3.0
only <https://spdx.org/licenses/GPL-3.0-only.html>`.

..
    Credits
    -------

    AIdsorb makes use of the following packages:

    .. image:: https://img.shields.io/badge/Numpy-black?logo=numpy
        :alt: Numpy
        :target: https://numpy.org

    .. image:: https://img.shields.io/badge/PyTorch-black?logo=pytorch
        :alt: PyTorch
        :target: https://pytorch.org/

    .. image:: https://img.shields.io/badge/PyTorch_Lightning-black?logo=lightning
        :alt: PyTorch Lightning
        :target: https://lightning.ai/docs/pytorch/stable/

    .. image:: https://img.shields.io/badge/Plotly-black?logo=plotly
        :alt: Plotly
        :target: https://plotly.com/

    .. image:: https://img.shields.io/badge/TQDM-black?logo=tqdm
        :alt: TQDM
        :target: https://tqdm.github.io/

    .. image:: https://img.shields.io/badge/Pandas-black?logo=pandas
        :alt: TQDM
        :target: https://pandas.pydata.org/

    .. image:: https://img.shields.io/badge/Python_Fire-black?logo=google
        :alt: Python Fire
        :target: https://google.github.io/python-fire/guide/

    .. image:: https://img.shields.io/badge/Mendeleev-black
        :alt: Mendeleev
        :target: https%3A%2F%2Fmendeleev.readthedocs.io%2Fen%2Fstable%2F

    .. image:: https://img.shields.io/badge/Atomic_Simulation_Environment-black
        :alt: Atomic Simulation Environment
        :target: https://wiki.fysik.dtu.dk/ase/index.html


.. toctree::
	:maxdepth: 1
	:caption: Contents

	installation
	tutorial
	cli
	api
	changelog


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
