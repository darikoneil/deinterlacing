import os
import sys
import toml
from unittest.mock import MagicMock
from pydantic.dataclasses import dataclass as PydanticDataclass

"""
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Mocks
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

# NOTE: Stops guarded type _typeshed warning during build that I can't seem to
#  permanently stamp out
MOCK_MODULES = ['_typeshed', 'PydanticDataclass']


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()



try:
    import cupy as cp
except ImportError:
    # NOTE: Read-the-docs doesn't have cupy installed, so we need to mock it
    MOCK_MODULES.append('cp')

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Path Setup and Package Details
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

# IMPORTS ps I can be done not so dumbly
sys.path.insert(0, os.path.dirname(os.path.dirname(os. getcwd())))

# Themes path
sys.path.append(os.path.abspath('_themes'))

# get package details directly from pyproject
pyproject_file = os.path.join(os.path.dirname(os.path.dirname(os. getcwd())), "pyproject.toml")
package_details = toml.load(pyproject_file).get("project")

project = package_details.get("name")
author = "Darik A. O'Neil"
#author = f"{package_details.authors}"  # f-string because maybe weird sphinx stuff if it gets list, not sure
release = package_details.get("version")


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Sphinx Configuration
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""
master_docs = 'index'

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinxcontrib.autodoc_pydantic',
    'sphinx_autodoc_typehints',
    'myst_parser',
]


templates_path = ['_templates']
exclude_patterns = []
language = 'en'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'cupy': ('https://docs.cupy.dev/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None)
}

root_doc = "index"

source_suffix = ".rst"

#html_theme_path = ['_themes']
html_theme = 'sphinx_rtd_theme'

pygments_style = "sphinx"

todo_include_todos = True

coverage_show_missing_items = True

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Sphinx Autodoc Typehints
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

typehints_defaults = 'comma'

always_document_param_types = True

typehints_fully_qualified = False

typehints_document_rtype = True

always_use_bars_union = True

simplify_optional_unions = False

typehints_use_signature = False

typehints_use_signature_return = False


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Autodoc Pydantic Configuration
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_config_summary = True
autodoc_pydantic_model_show_validator_summary = True
autodoc_pydantic_model_show_validator_members = True

"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Theme overrides
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

#html_css_files = ["css/custom.css"]

autodoc_mock_imports = [
    'cupy',
    'PydanticDataclass',
]
