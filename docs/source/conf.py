# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the package directory to the Python path
sys.path.insert(0, os.path.abspath('../../portfolio-lib-package'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'portfolio-lib'
copyright = '2025, Rahul Ashok, Pritham Devaprasad, Siddarth S, Anish R'
author = 'Rahul Ashok, Pritham Devaprasad, Siddarth S, Anish R'

version = '1.0.1'
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Master document (for compatibility)
master_doc = 'index'

# Language settings
language = 'en'

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'markdown',
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options for Read the Docs theme
html_theme_options = {
    'analytics_id': '',  # Provided by Read the Docs or Google Analytics
    'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS
html_css_files = [
    'custom.css',
]

# Page title format
html_title = f"{project} v{release} Documentation"

# Sidebar settings
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# Favicon (can be added later)
# html_favicon = '_static/favicon.ico'

# Logo (can be added later) 
# html_logo = '_static/logo.png'

# Show source link
html_show_sourcelink = True

# Copy source
html_copy_source = True

# Source link suffix
html_sourcelink_suffix = '.rst'

# Context for templates
html_context = {
    "display_github": True,  # Set to True to enable GitHub integration
    "github_user": "NeuralNinja110",  # Actual GitHub username
    "github_repo": "Portfolio-lib",  # Actual repo name
    "github_version": "main",  # Default branch
    "conf_py_path": "/docs/source/",
}

# -- Plot directive configuration --
plot_include_source = True
plot_html_show_source_link = False
plot_formats = ['png']
plot_html_show_formats = False
plot_basedir = os.path.abspath('.')
plot_rcparams = {
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.figsize': (8, 6),
    'figure.dpi': 100,
    'savefig.dpi': 150,
}

# -- Math configuration --
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}

# -- Autodoc configuration --
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Napoleon configuration --
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Intersphinx configuration --
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# -- Todo extension configuration --
todo_include_todos = False

# -- GitHub Pages configuration --
html_baseurl = 'https://neuralninja110.github.io/Portfolio-lib/'  # Actual GitHub Pages URL

# -- Read the Docs configuration --
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    html_theme = 'sphinx_rtd_theme'
    # RTD-specific settings
    html_context.update({
        "display_github": True,
        "github_user": "NeuralNinja110",  # Actual GitHub username
        "github_repo": "Portfolio-lib",  # Actual repo name
        "github_version": "main",
        "conf_py_path": "/docs/source/",
    })

# Suppress warnings
suppress_warnings = ['image.nonlocal_uri']

# Build settings
nitpicky = False
nitpick_ignore = [
    ('py:class', 'type'),
    ('py:class', 'optional'),
]
