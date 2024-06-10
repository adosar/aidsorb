:octicon:`rocket` Tutorial
==========================

.. note::
    **This tutorial covers the most common use cases of AIdsorb**. For more
    advanced usage, you should consult the :doc:`api`.

Introduction
------------

*What is a point cloud?*

    A point cloud is a *set of 3D coordinates* and (optionally) *associated
    features*. More formally:

    .. math::
        \mathcal{P} = \{
            \mathbf{p}_1, \mathbf{p}_2, \dotsc, \mathbf{p}_3
        \}_{i=1}^N
        \quad
        \text{and}
        \quad
        \mathbf{p}_i \in \mathbb{R}^{3+C}

    where :math:`N` is the number of points in the cloud and :math:`C` is the
    number of features.

    In AIdsorb, a point cloud is represented as a :class:`numpy.ndarray` of
    shape ``(N, 3+C)``:

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

    In AIdsorb, a molecular ``pcd`` is represented as :class:`numpy.ndarray` of
    shape ``(N, 4+C)``, where ``N`` is the number of atoms, ``pcd[:, :3]`` are the
    **atomic coordinates**, ``pcd[:, 3]`` are the **atomic numbers** and ``pcd[:,
    4:]`` any **additional features**. If ``C == 0``, then the only features are the
    atomic numbers.


Create and visualize a molecular point cloud
--------------------------------------------

For this part of the tutorial: :download:`IRMOF-1.xyz
<../../tests/structures/IRMOF-1.xyz>`.

Create a molecular point cloud::

    >>> from aidsorb.utils import pcd_from_file
    >>> mol_name, pcd = pcd_from_file('path/to/IRMOF-1.xyz')

Visualize it:

.. tab-set::

    .. tab-item:: Python
        
        .. code-block::

            >>> from aidsorb.visualize import draw_pcd
            >>> draw_pcd(pcd)

    .. tab-item:: CLI
        
        .. code-block:: console

            $ aidsorb visualize path/to/IRMOF-1.xyz


..
    .. raw:: html
        :file: images/pcd_plotly.html

Hover your pointer :fa:`arrow-pointer; fa-beat-fade` over the figure to play with it!

Machine learning on molecular point clouds
------------------------------------------

For this part of the tutorial: :download:`toy_project.zip
<download/toy_project.zip>`.

For performing machine learning with AIdsorb, the following components are
needed:

* A directory containing files of **molecular structures**.
* A ``.csv`` file containing the **labels of the molecular structures**.
* A ``.yaml`` **configuration file** for orchestrating the training part.

*and you are solely responsible for them*.

After downloading and unzipping the file, you will get a directory structure
populated with these 3 components:

.. code-block:: console

    $ tree toy_project
    toy_project
    ├── configs
    │   └── config_example.yaml
    ├── labels
    │   └── labels.csv
    ├── pcd_data  # This directory will be populated later by AIdsorb.
    └── structures
        ├── ala_phe_ala.pdb
        ├── COF-5.cif
        └── ...

.. note::
    **The above directory structure is not required by AIdsorb**. It is just there
    for illustration purposes only, to help you organize better your project.

Prepare the data
^^^^^^^^^^^^^^^^

Create the point clouds:

    .. tab-set::
    
        .. tab-item:: CLI

            .. code-block:: console

                $ aidsorb create toy_project/structures toy_project/pcd_data/point_clouds.npz -f "['en_pauling']"

        .. tab-item:: Python

            .. code-block:: python

                from aidsorb.utils import pcd_from_dir

                # We add electronegativity as additional feature.
                pcd_from_dir(
                    dirname='toy_project/structures',
                    outname='toy_project/pcd_data/point_clouds.npz',
                    features=['en_pauling'],
                )

Split the point clouds into train, validation and test sets:

    .. tab-set::
    
        .. tab-item:: CLI

            .. code-block:: console

                $ aidsorb prepare toy_project/pcd_data/point_clouds.npz --split_ratio "(0.33

        .. tab-item:: Python

            .. code-block:: python

                from aidsorb.utils import pcd_from_dir

                # We add electronegativity as additional feature.
                pcd_from_dir(
                    dirname='toy_project/structures',
                    outname='toy_project/pcd_data/point_clouds.npz',
                    features=['en_pauling'],
                )
..
    Can I use other point clouds
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Train the algorithm
    ^^^^^^^^^^^^^^^^^^^

    Can I use vanilla PyTorch?
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    Add this later with faking retarded rst sucking dicks.

    Add a link to :download:`IRMOF-1.xyz <../../tests/samples/IRMOF-1.xyz>`

    ADd an octicon :fab:`python;fa-lg fa-bounce`.

    .. toctree::
        :maxdepth: 2
        :name: Put a faking name to use a role

        download/fooba
