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
molecular point clouds**.

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

               .. button-link:: https://example.com
                       :color: primary
                       :align: center

                       📦 PyPI package

       .. grid-item::

               .. button-link:: https://example.com
                       :color: primary
                       :align: center

                       :octicon:`telescope` |aidsorb| paper

|aidsorb| aims to provide a **simple, easy-to-use and reproduce** interface
for:

* 📥 **Creating molecular point clouds**
* 🤖 **Training DL algorithms on molecular point clouds**

.. raw:: html
    :file: images/pcd_plotly.html

.. rubric:: Why molecular point clouds?

A *molecular point cloud*, being essentially a *set of atomic positions, atomic
numbers and any additional information*, provides a natural and efficient way to
represent molecular structures and chemical systems in a machine understandable
format. This in turns allows us to **perform DL directly on raw structural
information**, removing the need for manual feature extraction.

The above point cloud represents IRMOF-1. You can hover :fa:`arrow-pointer;
fa-beat-fade` over the figure to play with it.

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

        This might require the usage of |pyg|.

    .. card:: 4️⃣  Extend featurization
        :text-align: center

        Add more featurization options. These should be fast!

Contributing
------------

🙌 We welcome contributions from the community to help improve and expand this
project!

You can start by 🛠️ `opening an issue <https://github.com/adosar/aidsorb/issues>`_ for:

* 🐛 Reporting bugs
* 🌟 Suggesting new features
* 📚 Improving documentation
* 🎨 Adding your example to the :doc:`Gallery <auto_examples/index>`

We appreciate your efforts to submit well-documented :octicon:`git-pull-request`
`pull requests <https://github.com/adosar/aidsorb/pulls>`_ and participate in
discussions.

💪 Together, we can make this project even better!

Citing
------

If you use AIdosrb in your research, please consider citing the following work::

	Currently, not available.


License
-------

|aidsorb| is released under the |license|.

.. toctree::
    :maxdepth: 1
    :caption: Contents

    installation
    tutorial
    cli
    api
    auto_examples/index
    changelog


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
