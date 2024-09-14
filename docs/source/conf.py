# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys, os
from importlib.metadata import version as get_version
from plotly.io._sg_scraper import plotly_sg_scraper

sys.path.insert(0, os.path.abspath('../../src'))

project = 'AIdsorb'
copyright = '2024, Antonios P. Sarikas'
author = 'Antonios P. Sarikas'
release = get_version('aidsorb')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'sphinx.ext.todo',
        'sphinx.ext.autosectionlabel',
        'sphinx.ext.intersphinx',
        'sphinx.ext.viewcode',
        'sphinx.ext.autosummary',
        'sphinx_design',
        'sphinx_copybutton',
        'sphinx_gallery.gen_gallery',
        ]

copybutton_exclude = '.linenos, .gp, .go'
todo_include_todos = True
autodoc_inherit_docstrings = False
autodoc_typehints = 'none'
autosectionlabel_prefix_document = True

templates_path = ['_templates']
#exclude_patterns = ['_autosummary/*']

sphinx_gallery_conf = {
         'examples_dirs': 'examples',  # path to your example scripts
         'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
         'ignore_pattern': r'{/.*\.xyz, /.*\.cif}',
         'image_scrapers': (plotly_sg_scraper),
         }

intersphinx_mapping = {
        'python': ('https://docs.python.org/3', None),
        'numpy': ('https://numpy.org/doc/stable/', None),
        'pytorch': ('https://pytorch.org/docs/stable', None),
        'lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
        'torchmetrics': ('https://lightning.ai/docs/torchmetrics/stable/', None),
        'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
        'plotly': ('https://plotly.com/python-api-reference/', None),
        }

# For commonly used links and inline text.
rst_epilog = '''
.. |pytorch| replace:: :bdg-link-primary:`PyTorch <https://pytorch.org/>`
.. |lightning| replace:: :bdg-link-primary:`PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`
.. |pyg| replace:: :bdg-link-primary:`PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`
.. |license| replace:: :bdg-link-primary:`GNU General Public License v3.0 only <https://spdx.org/licenses/GPL-3.0-only.html>`
.. |aidsorb| replace:: **AIdsorb**
'''

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'renku'
html_static_path = ['_static']
html_logo = 'images/aidsorb_logo_dark.svg'
html_theme_options = {'logo_only': True}
html_css_files = ["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"]
