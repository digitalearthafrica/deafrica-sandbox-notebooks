<img align="centre" src="../Supplementary_data/Github_banner.jpg" width="100%">

# DE Africa Tools Package


This folder stores the dependecies for installing the python package `deafrica_tools`, which contains a set of python functions and algorithms developed to assist in analysing DE Africa data (e.g. loading data, plotting, spatial analysis)

The package is being managed by [poetry](https://python-poetry.org/)


## How to build the package

Install poetry first.

In the repo root, run:

```bash
poetry build
```

config the testpypi by following command:

```bash
poetry config repositories.test https://test.pypi.org/legacy/
```

Then you are required to fill in your user name and password.

One published testing is on testpypi [dea-common](https://test.pypi.org/project/dea-common/)

## publish the repo

Use testpypi to test the releae according to the tutorial on [using-testpypi](https://packaging.python.org/guides/using-testpypi/).

You can search you packages on: [test.pypi.org](https://test.pypi.org/manage/projects/).

## install the package from local build

After run poetry build, you can install the deafrica_tools as bellow

```bash
pip install -e dist/deafrica_tools_<version>-none-any.whl
```
