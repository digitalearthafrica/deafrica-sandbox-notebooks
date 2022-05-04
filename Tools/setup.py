#!/usr/bin/env python3

import io
import os
from setuptools import find_packages, setup

# Where are we?
IS_DEAFRICA_SANDBOX = ('sandbox' in os.getenv('JUPYTER_IMAGE', default=''))

# What packages are required for this module to be executed?
# These are all on the Sandbox so shouldn't need installing on those platforms.
REQUIRED = [
    # load_era5
    'fsspec',
    # classification
    'numpy',
    'xarray',
    'geopandas',
    'datacube',
    'tqdm',
    'dask',
    'rasterio',
    'scikit-learn',
    # coastal
    'matplotlib',
    'pandas',
    'scipy',
    # 'otps',  # Hard to install, but available on Sandbox
    # datahandling
    'GDAL',
    'odc-ui',
    'numexpr',
    # plotting
    'folium',
    'pyproj',
    'branca',
    'shapely',
    'scikit-image',
    # temporal
    'hdstats',
    # spatial
    'rasterstats',
    'geopy',
    'OWSLib',
    'fiona',
    'shapely',
    # app subpackage modules
    'datetime'
]

# What packages are optional?
EXTRAS = {
    'jupyter': ['IPython', 'ipywidgets', 'ipyleaflet'],
    'boto': ['boto3'],
}

# Package meta-data.
NAME = 'deafrica-tools'
DESCRIPTION = 'Functions and algorithms for analysing Digital Earth Africa data.'
URL = 'https://github.com/digitalearthafrica/deafrica-sandbox-notebooks'
EMAIL = 'systems@digitalearthafrica.org'
AUTHOR = 'Digital Earth Africa'
REQUIRES_PYTHON = '>=3.6.0'    

# Import the README and use it as the long-description.
here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION
    
setup_kwargs = {
    'name': NAME,
    'version': '0.1.2',
    'description': DESCRIPTION,
    'long_description': long_description,
    'author': AUTHOR,
    'author_email': EMAIL,
    'python_requires': REQUIRES_PYTHON,
    'url': URL,
    'install_requires': REQUIRED if not IS_DEAFRICA_SANDBOX else [],
    'packages': find_packages(),
    'include_package_data':True,
    'license':'Apache License 2.0',
    'package_data': {'': ['locales/*/*/*.mo', 'locales/*/*/*.po']},
}

setup(**setup_kwargs)
