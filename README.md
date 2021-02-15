
<img align="centre" src="Supplementary_data/Github_banner.jpg" width="100%">

# Digital Earth Africa Notebooks

<img align="left" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">.


**License:** The code in this repository is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). Digital Earth Africa data is licensed under the [Creative Commons by Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).

**Contact:** If you need assistance with any of the Jupyter Notebooks or Python code in this repository, please post a question on the [Open Data Cube Slack channel](http://slack.opendatacube.org/) or on the [GIS Stack Exchange](https://gis.stackexchange.com/questions/ask?tags=open-data-cube) using the `open-data-cube` tag (you can view `previously asked questions` [here](https://gis.stackexchange.com/questions/tagged/open-data-cube). If you would like to report an issue with this notebook, you can file one on the [Github issues page](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues)

---

The Digital Earth Africa Notebooks repository (`deafrica-sandbox-notebooks`) hosts Jupyter Notebooks, Python scripts and workflows for analysing [Digital Earth Africa (DE Africa)](https://www.digitalearthafrica.org/) satellite data and derived products. This documentation is designed to provide a guide to getting started with DE Africa, and to showcase the wide range of geospatial analyses that can be achieved using DE Africa data and open-source software including [Open Data Cube](https://www.opendatacube.org/) and [xarray](http://xarray.pydata.org/en/stable/).

The repository is based around the following directory structure (from simple to increasingly complex applications):

1. [Beginners_guide](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/master/Beginners_guide): *Introductory notebooks aimed at introducing Jupyter Notebooks and how to load, plot and interact with DE Africa data*

2. [Datasets](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/master/Datasets): *Notebooks introducing DE Africa's satellite datasets and derived products, including how to load each dataset and any special features of the data. Some external datasets that are useful for analysing and interpreting DE Africa products are also covered.*

3. [Frequently_used_code](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/master/Frequently_used_code): *A recipe book of simple code examples demonstrating how to perform common geospatial analysis tasks using DE Africa and open-source software*

4. [Real_world_examples](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/master/Real_world_examples): *More complex workflows demonstrating how DE Africa can be used to address real-world problems*

The supporting scripts and data for the notebooks are kept in the following directories:

- [Scripts](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/master/Scripts): *Python functions and algorithms developed to assist in analysing DE Africa data (e.g. loading data, plotting, spatial analysis, machine learning)* 

- [Supplementary_data](https://github.com/GeoscienceAustralia/dea-notebooks/tree/master/Supplementary_data): *Supplementary files required for the analyses above (e.g. images, rasters, shapefiles, training data)*


---

## Getting started with DE Africa Notebooks


To get started with using `deafrica-sandbox-notebooks`, visit the DE Africa Notebooks [Wiki page](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/wiki). This page includes guides for getting started on the [DE Africa Sandbox](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/wiki#getting-started-on-the-de-africa-sandbox).

Once you're set up, the main option for interacting with `deafrica-sandbox-notebooks` and contributing back to the repository is through:

* **DE Africa notebooks using git**: Git is a version-control software designed to help track changes to files and collaborate with multiple users on a project. Using ``git`` is the recommended workflow for working with ``deafrica-sandbox-notebooks`` as it makes it easy to stay up to date with the latest versions of functions and code, and makes it impossible to lose your work. 

  * Refer to the repository's [Guide to using DE Africa Notebooks with git](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/wiki/Guide-to-using-DE-Africa-Notebooks-with-git) wiki article.

---

## Contributing to DE Africa Notebooks

### Master and working branches

The `deafrica-sandbox-notebooks` repository uses 'branches' to manage individuals' notebooks, and to allow easy publishing of notebooks ready to be shared. There are two main types of branches:

* [Master branch](https://github.com/GeoscienceAustralia/dea-notebooks/tree/master): The ``master`` branch contains DE Africa's collection of publicly available notebooks. The ``master`` branch is protected, and is only updated after new commits a reviewed and approved by the DE Africa team.
* [Working branches](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/branches): All other branches in the repository are working spaces for users of ``deafrica-sandbox-notebooks``. They have a unique name (typically named after the user). The notebooks on these branches can be works-in-progress and do not need to be pretty or complete. By using a working branch, it is easy to use scripts and algorithms from ``deafrica-sandbox-notebooks`` in your own work, or share and collaborate on a working version of a notebook or code.

--- 
### Publishing notebooks to the master branch


Once you have a notebook that is ready to be published on the ``master`` branch, you can submit a 'pull request' in the [Pull requests tab at the top of the repository](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/pulls). The default pull request template contains a check-list to ensure that all ``master`` branch Jupyter notebooks are consistent and well-documented so they can be understood by future users, and rendered correctly. Please ensure that as many of these checklist items are complete as possible, or leave a comment in the pull request asking for help with any remaining checklist items.

#### Draft pull requests

For pull requests you would like help with or that are a work in progress, consider using Github's [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/) feature. This indicates that your work is still a draft, allowing you to get feedback from other DE Africa users before it is published on the `master` branch.

---
### DE Africa Notebooks template notebook

A template notebook has been developed to make it easier to create new notebooks that meet all the pull request checklist requirements. The template notebook contains a simple structure and useful general advice on writing and formatting Jupyter notebooks. The template can be found here: [DEAfrica_notebooks_template.ipynb](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/blob/master/DEAfrica_notebooks_template.ipynb)

Using the template is not required for working branch notebooks, but is *highly recommended* as it will make it much easier to publish any notebooks on ``master`` in the future.

---
### Approving pull requests

Anyone with admin access to the ``deafrica-sandbox-notebooks`` repository can approve 'pull requests'.

If the notebook meets all the checklist requirements, click the green 'Review' button and click 'Approve' (with an optional comment). You can also 'Request changes' here if any of the checklist items are not complete.

Once the pull request has been approved, you can merge it into the ``master`` branch. Select the 'Squash and merge' option from the drop down menu to the right of the green 'merge' button. Once you have merged the new branch in, you need to delete the branch. There is a button on the page that asks you if you would like to delete the now merged branch. Select 'Yes' to delete it.

