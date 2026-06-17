# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
from importlib.metadata import version as get_version

from plotly.io._sg_scraper import plotly_sg_scraper

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
        'sphinx.ext.extlinks',
        'sphinx_design',
        'sphinx_copybutton',
        'sphinx_gallery.gen_gallery',
        ]

copybutton_exclude = '.linenos, .gp, .go'
todo_include_todos = True
autodoc_inherit_docstrings = False
autodoc_typehints = 'description'
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

extlinks = {'issue': ('https://github.com/adosar/aidsorb/issues/%s', '#%s')}

# For commonly used links and inline text.
rst_epilog = '''
.. |pytorch| replace:: :bdg-link-light:`PyTorch <https://pytorch.org/>`
.. |lightning| replace:: :bdg-link-light:`PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`
.. |license| replace:: :bdg-link-light:`GNU General Public License v3.0 only <https://spdx.org/licenses/GPL-3.0-only.html>`
.. |aidsorb| replace:: :bdg-link-light:`AIdsorb <https://github.com/adosar/aidsorb/>`
.. |github| replace:: :bdg-link-light:`GitHub <https://github.com/adosar/aidsorb/>`
.. |discussions| replace:: :bdg-link-light:`Discussions <https://github.com/adosar/aidsorb/discussions>`
.. |contributing-guide| replace:: :bdg-link-light:`Contributing Guide <https://github.com/adosar/aidsorb?tab=contributing-ov-file>`
'''

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_theme = 'pydata_sphinx_theme'
#html_static_path = ['_static']
#html_logo = 'images/aidsorb_logo_dark.svg'
html_context = {'default_mode': 'light'}

html_theme_options = {
        'header_links_before_dropdown': 4,
        'logo': {
            #'text': 'Documentation',
            'alt_text': 'AIdsorb documentation - Home',
            'image_light': 'images/aidsorb_logo_light.svg',
            'image_dark': 'images/aidsorb_logo_dark.svg',
            },
        'icon_links': [
            {
                'name': 'GitHub',
                'url': 'https://github.com/adosar/aidsorb',
                'icon': 'fa-brands fa-github',
                'type': 'fontawesome',
                },
            {
                'name': 'PyPI',
                'url': 'https://pypi.org/project/aidsorb/',
                'icon': 'fa-brands fa-python',
                'type': 'fontawesome',
                },
            {
                'name': 'Paper',
                'url': 'https://www.nature.com/articles/s41598-024-76319-8',
                'icon': 'fa-solid fa-microscope',
                'type': 'fontawesome',
                },
            ]
        }
#html_theme_options = {'logo_only': True}
#html_css_files = [
#        'custom.css',
#        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css',
#        ]
