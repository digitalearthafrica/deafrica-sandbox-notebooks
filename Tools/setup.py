#!/usr/bin/env python3

import os
from pathlib import Path
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

# Set the value of long_description to the contents (not the path) of the README file itself.
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
# Set the long_description_content_type to an accepted Content-Type-style value for your README file’s markup.
long_description_content_type='text/markdown'
    
setup_kwargs = {
    'name': NAME,
    'version': '0.1.4',
    'description': DESCRIPTION,
    'long_description': long_description,
    'long_description_content_type' : long_description_content_type,
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
