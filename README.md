# Python Development Template Repository
Simply a template repository for development in python. While it contains a pyproject.toml, this template is not 
restricted to use in the development of python packages. You can perform an 'editable install' of your local code to 
leverage the benefits in a more transient scripting or analysis environment. 

```
python -m pip install -e .
```

## Features
- [x] Package template file with pyproject.toml
- [x] Formatting with isort & black
- [x] Linting with flake8 & various plugins
- [x] Testing with pytest & various plugins
- [x] Coverage with pycoverage & coveralls
- [x] Documentation with sphinx, read-the-docs, & various plugins
- [x] Scripts for automation of code review, documentation, and distribution
- [x] GitHub Actions templates for post-commit linting, coverage, and cross-platform testing

## Installation
To install extra dependencies for development, install your package with one or all of the desired optional 
dependencies.

```
python -m pip install -e .[testing]
python -m pip install -e .[linting]
python -m pip install -e .[coverage]
python -m pip install -e .[distribution]
python -m pip install -e .[documentation]
python -m pip install -e .[all]
```
