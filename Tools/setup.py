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
VERSION = '0.1.10'
DESCRIPTION = 'Functions and algorithms for analysing Digital Earth Africa data.'
# Set the value of long_description to the contents (not the path) of the README file itself.
this_directory = Path(__file__).parent
LONG_DESCRIPTION = (this_directory / 'README.md').read_text()
# Set the long_description_content_type to an accepted Content-Type-style value for your README file’s markup.
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
AUTHOR = 'Digital Earth Africa'
AUTHOR_EMAIL = 'systems@digitalearthafrica.org'
REQUIRES_PYTHON = '>=3.6.0'    
URL = 'https://github.com/digitalearthafrica/deafrica-sandbox-notebooks'
BUG_TRACKER_URL = 'https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues' 


    
setup_kwargs = {
    'name': NAME,
    'version': VERSION,
    'description': DESCRIPTION,
    'long_description': LONG_DESCRIPTION,
    'long_description_content_type' : LONG_DESCRIPTION_CONTENT_TYPE,
    'author': AUTHOR,
    'author_email': AUTHOR_EMAIL,
    'python_requires': REQUIRES_PYTHON,
    'url': URL,
    'project_urls' : {'Bug Tracker': BUG_TRACKER_URL},
    'install_requires': REQUIRED if not IS_DEAFRICA_SANDBOX else [],
    'packages': find_packages(),
    'include_package_data':True,
    'license':'Apache License 2.0',
    'package_data': {'': ['locales/*/*/*.mo', 'locales/*/*/*.po']},
}

setup(**setup_kwargs)
