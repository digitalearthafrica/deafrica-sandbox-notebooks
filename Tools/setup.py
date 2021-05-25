# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['deafrica_tools']

package_data = \
{'': ['*']}

install_requires = \
['dask-ml>=1.8.0,<2.0.0',
 'datacube>=1.8.3,<2.0.0',
 'geopandas>=0.8.2,<0.9.0',
 'joblib>=1.0.1,<2.0.0',
 'matplotlib>=3.3.4,<4.0.0',
 'scikit-learn>=0.24.1,<0.25.0',
 'tqdm>=4.57.0,<5.0.0']

setup_kwargs = {
    'name': 'deafrica-tools',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'Digital Earth Africa Team',
    'author_email': None,
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/digitalearthafrica/dea-common.git',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.6,<4.0',
}


setup(**setup_kwargs)
