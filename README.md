
<img align="centre" src="Supplementary_data/Github_banner.jpg" width="100%">

# Digital Earth Africa Notebooks

<img align="left" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"> </br>

**License:** The code in this repository is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). Digital Earth Africa data is licensed under the [Creative Commons by Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).

**Contact:** If you need assistance with any of the Jupyter Notebooks or Python code in this repository, please post a question on the [Open Data Cube Slack channel](http://slack.opendatacube.org/) or on the [GIS Stack Exchange](https://gis.stackexchange.com/questions/ask?tags=open-data-cube) using the `open-data-cube` tag (you can view `previously asked questions` [here](https://gis.stackexchange.com/questions/tagged/open-data-cube). If you would like to report an issue with any of the scripts or notebooks in this repository, you can file one on the [Github issues page](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues).

**Citing DE Africa Notebooks:** If you use any of the notebooks, code or tools in this repository in your work, please reference them using the following citation:

    Krause, C., Dunn, B., Bishop-Taylor, R., Adams, C., Burton, C., Alger, M., Chua, S., Phillips, C., Newey, V., Kouzoubov, K., Leith, A., Ayers, D., Hicks, A., DEA Notebooks contributors 2021. Digital Earth Australia notebooks and tools repository. Geoscience Australia, Canberra. https://doi.org/10.26186/145234

---

The Digital Earth Africa Notebooks repository (`deafrica-sandbox-notebooks`) hosts Jupyter Notebooks, Python scripts and workflows for analysing [Digital Earth Africa (DE Africa)](https://www.digitalearthafrica.org/) satellite data and derived products. This documentation is designed to provide a guide to getting started with DE Africa, and to showcase the wide range of geospatial analyses that can be achieved using DE Africa data and open-source software including [Open Data Cube](https://www.opendatacube.org/) and [xarray](http://xarray.pydata.org/en/stable/).

The repository is based around the following directory structure (from simple to increasingly complex applications):

1. [Beginners_guide](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/main/Beginners_guide): *Introductory notebooks aimed at introducing Jupyter Notebooks and how to load, plot and interact with DE Africa data.*

2. [Datasets](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/main/Datasets): *Notebooks introducing DE Africa's satellite datasets and derived products, including how to load each dataset and any special features of the data. Some external datasets that are useful for analysing and interpreting DE Africa products are also covered.*

3. [Frequently_used_code](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/main/Frequently_used_code): *A recipe book of simple code examples demonstrating how to perform common geospatial analysis tasks using DE Africa and open-source software.*

4. [Real_world_examples](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/main/Real_world_examples): *More complex workflows demonstrating how DE Africa can be used to address real-world problems.*

5. [Use Cases](https://github.com/GeoscienceAustralia/dea-notebooks/tree/main/Use_cases): *Notebooks in this collection are developed for specific use-cases of the Digital Earth Africa platform and may not run as seamlessly as notebooks in the other folders of this repository. Notebooks may contain less descriptive markdown, contain more complicated or bespoke analysis, and may take a long time to run. However, they contain useful analysis procedures and provide further examples for advanced users.*

The supporting scripts and data for the notebooks are kept in the following directories:

- [Tools](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/main/Tools): *Python functions and algorithms developed to assist in analysing DE Africa data (e.g. loading data, plotting, spatial analysis, machine learning).* 

- [Supplementary_data](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/main/Supplementary_data): *Supplementary files required for the analyses above (e.g. images, rasters, shapefiles, training data).*

---

## Getting started with DE Africa Notebooks

To get started with using `deafrica-sandbox-notebooks`, visit 
the DE Africa Notebooks [Wiki page](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/wiki). This page includes guides for getting started on the [DE Africa Sandbox](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/wiki#getting-started-on-the-de-africa-sandbox).

Once you're set up, the main option for interacting with `deafrica-sandbox-notebooks` and contributing back to the repository is through:

* **DE Africa notebooks using Git:** Git is a version-control software designed to help track changes to files and collaborate with multiple users on a project. Using ``git`` is the recommended workflow for working with ``deafrica-sandbox-notebooks`` as it makes it easy to stay up to date with the latest versions of functions and code, and makes it impossible to lose your work. 

  * Refer to the repository's [Guide to using DE Africa Notebooks with git](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/wiki/Guide-to-using-DE-Africa-Notebooks-with-git) wiki article. For a more detailed explanation suited to new Git users, see our [Version Control with Git](https://docs.digitalearthafrica.org/en/latest/sandbox/git-howto/index.html) tutorial.
  
* **Set up Git authentication tokens:** Git requires multi-factor authentication when using the command line or API. Set up a personal access token by following instructions from the [GitHub Docs](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

---

## Contributing to DE Africa Notebooks

### Main and working branches

The `deafrica-sandbox-notebooks` repository uses 'branches' to manage individuals' notebooks, and to allow easy publishing of notebooks ready to be shared. There are two main types of branches:

* [Main branch](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/main): The ``main`` branch contains DE Africa's collection of publicly available notebooks. The ``main`` branch is protected, and is only updated after new commits a reviewed and approved by the DE Africa team.
* [Working branches](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/branches): All other branches in the repository are working spaces for users of ``deafrica-sandbox-notebooks``. They have a unique name (typically named after the user). The notebooks on these branches can be works-in-progress and do not need to be pretty or complete. By using a working branch, it is easy to use scripts and algorithms from ``deafrica-sandbox-notebooks`` in your own work, or share and collaborate on a working version of a notebook or code.

--- 
### Publishing notebooks to the main branch


Once you have a notebook that is ready to be published on the ``main`` branch, you can submit a 'pull request' in the [Pull requests tab at the top of the repository](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/pulls). The default pull request template contains a check-list to ensure that all ``main`` branch Jupyter notebooks are consistent and well-documented so they can be understood by future users, and rendered correctly. Please ensure that as many of these checklist items are complete as possible, or leave a comment in the pull request asking for help with any remaining checklist items.

#### Draft pull requests

For pull requests you would like help with or that are a work in progress, consider using Github's [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/) feature. This indicates that your work is still a draft, allowing you to get feedback from other DE Africa users before it is published on the `main` branch.

---
### DE Africa Notebooks template notebook

A template notebook has been developed to make it easier to create new notebooks that meet all the pull request checklist requirements. The template notebook contains a simple structure and useful general advice on writing and formatting Jupyter notebooks. The template can be found here: [DEAfrica_notebooks_template.ipynb](https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/blob/main/DEAfrica_notebooks_template.ipynb)

Using the template is not required for working branch notebooks, but is *highly recommended* as it will make it much easier to publish any notebooks on ``main`` in the future.

---
### Approving pull requests

Anyone with admin access to the ``deafrica-sandbox-notebooks`` repository can approve 'pull requests'.

If the notebook meets all the checklist requirements, click the green 'Review' button and click 'Approve' (with an optional comment). You can also 'Request changes' here if any of the checklist items are not complete.

Once the pull request has been approved, you can merge it into the ``main`` branch. Select the 'Squash and merge' option from the drop down menu to the right of the green 'merge' button. Once you have merged the new branch in, you need to delete the branch. There is a button on the page that asks you if you would like to delete the now merged branch. Select 'Yes' to delete it.

---

## Update: The default branch has been renamed!
*October 2021*

``master`` is now named ``main`` in line with GitHub recommended naming conventions.

If you have a local clone created before 29 October 2021, you can update it by running the following commands.

```
git branch -m master main
git fetch origin
git branch -u origin/main main
git remote set-head origin -a
```
