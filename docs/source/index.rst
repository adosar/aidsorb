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

|aidsorb| is a :fa:`python; fa-fade` Python package for **deep learning on
porous materials**.

.. grid:: 4

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

            .. button-link:: https://pypi.org/project/aidsorb/
                  :color: primary
                  :align: center

                  📦 PyPI package

      .. grid-item::

            .. button-link:: https://www.nature.com/articles/s41598-024-76319-8
                  :color: primary
                  :align: center

                  :octicon:`telescope` |aidsorb| paper

|aidsorb| aims to provide a **simple, easy-to-use and reproduce** interface
for:

* 📥 **Creating add text**
* 🤖 **Training add text**

Add text and wave snippets.

TODO
----

.. card-carousel:: 2

   .. card:: 1️⃣  Extend the :doc:`cli`
      :text-align: center

      Enable users to make predictions from the command line.

   .. card:: 2️⃣  Add pretrained models
      :text-align: center

      Enable users to fine-tune models trained on large data.

   .. card:: 3️⃣  Support more architectures
      :text-align: center

      Provide clean and fast implementations of new architectures.

   .. card:: 4️⃣  Extend featurization
      :text-align: center

      Add more featurization options. These should be fast!

Contributing
------------

We welcome contributions from the community! Please read our |contributing-guide| before submitting PRs or opening issues.

Citing
------

.. tab-set::

    .. tab-item:: Software

       Please refer to the `citation file <https://github.com/adosar/aidsorb/blob/master/CITATION.cff>`_
       or click the citation button on |github|.

    .. tab-item:: Paper

        .. code-block:: bibtex

            @article{Sarikas2024,
              title = {Gas adsorption meets geometric deep learning: points, set and match},
              volume = {14},
              ISSN = {2045-2322},
              url = {http://dx.doi.org/10.1038/s41598-024-76319-8},
              DOI = {10.1038/s41598-024-76319-8},
              number = {1},
              journal = {Scientific Reports},
              publisher = {Springer Science and Business Media LLC},
              author = {Sarikas,  Antonios P. and Gkagkas,  Konstantinos and Froudakis,  George E.},
              year = {2024},
              month = nov
            }


License
-------

|aidsorb| is released under the |license|.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   installation
   getting_started
   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   cli
   changelog


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
