<img align="centre" src="../Supplementary_data/Github_banner.jpg" width="100%">

# DE Africa Tools Package


Python functions and algorithms for developed to assist in analysing Digital Earth Africa data (e.g. loading data, plotting, spatial analysis, machine learning).

Installation
------------

To work with this module on the Digital Earth Africa Sandbox from within the `deafrica-sandbox-notebooks` repo, you can add the Tools folder to the system path:

       import sys
       sys.path.insert(1, '../Tools/')
       import dea_tools.datahandling  # or some other submodule

This module is automatically installed on the Sandbox. If for some reason the module isn't avilable, you can also `pip install` the module.
To do this on the Digital Earth Africa Sandbox, run `pip` from the terminal:

       pip install -e Tools/

To install this module from the source on any other system with `pip`:

    pip install --extra-index-url="https://packages.dea.ga.gov.au" git+https://github.com/digitalearthafrica/deafrica-sandbox-notebooks.git#subdirectory=Tools


Citing DE Africa Tools
----------------------

The code in this module is an adaptation of code from the [Digital Earth Australia](https://github.com/GeoscienceAustralia/dea-notebooks) `dea-tools` package. If you use any of the code in this repository in your work, please reference them using the following citation:

    Krause, C., Dunn, B., Bishop-Taylor, R., Adams, C., Burton, C., Alger, M., Chua, S., Phillips, C., Newey, V., Kouzoubov, K., Leith, A., Ayers, D., Hicks, A., DEA Notebooks contributors 2021. Digital Earth Australia notebooks and tools repository. Geoscience Australia, Canberra. https://doi.org/10.26186/145234