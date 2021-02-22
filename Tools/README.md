<img align="centre" src="../Supplementary_data/Github_banner.jpg" width="100%">

# DE Africa Tools Package


This folder stores the dependecies for installing the python package `deafrica_tools`, which contains a set of python functions and algorithms developed to assist in analysing DE Africa data (e.g. loading data, plotting, spatial analysis)

The package is being managed by [poetry](https://python-poetry.org/)


## How to build & install the package

In the terminal, first install poetry:

```pip install poetry```

Then navigate to the `"deafrica-sandbox-notebooks/Tools"` folder, and run:

```poetry build```

After the poetry build, you can install the `deafrica_tools` package by running:

```pip install dist/deafrica_tools-0.1.0.tar.gz```


## Configuring the testpypi:

```bash
poetry config repositories.test https://test.pypi.org/legacy/
```

Then you are required to fill in your user name and password.

One published testing is on testpypi [dea-common](https://test.pypi.org/project/dea-common/)

## Publishing the repo

Use testpypi to test the release according to the tutorial on [using-testpypi](https://packaging.python.org/guides/using-testpypi/).

You can search you packages on: [test.pypi.org](https://test.pypi.org/manage/projects/).

