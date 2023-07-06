DE Africa Tools Package
=======================

``deafrica_tools`` is a Python package contains several modules with functions to load, analyse
and output data from Digital Earth Africa. It is automatically installed in the Digital Earth 
Africa Sandbox environment. More information on installing this package can be found on the `Tools
<https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/master/Tools/>`_ section of the GitHub repository.

Core modules
-----------------

.. autosummary::
   :toctree: gen

   deafrica_tools.areaofinterest
   deafrica_tools.bandindices
   deafrica_tools.classification
   deafrica_tools.coastal
   deafrica_tools.dask
   deafrica_tools.datahandling
   deafrica_tools.load_era5
   deafrica_tools.load_isda
   deafrica_tools.load_soil_moisture
   deafrica_tools.plotting
   deafrica_tools.spatial
   deafrica_tools.temporal
   deafrica_tools.wetlands

Apps and widgets
-----------------

``deafrica_tools`` app subpackages can be accessed through ``deafrica_tools.app``.

.. autosummary::
   :toctree: gen
   
   deafrica_tools.app.animations
   deafrica_tools.app.changefilmstrips
   deafrica_tools.app.crophealth
   deafrica_tools.app.deacoastlines
   deafrica_tools.app.forestmonitoring
   deafrica_tools.app.geomedian
   deafrica_tools.app.imageexport
   deafrica_tools.app.wetlandsinsighttool
   deafrica_tools.app.widgetconstructors

License
-------
The code in this module is licensed under the Apache License,
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

Digital Earth Africa data is licensed under the Creative Commons by
Attribution 4.0 license (https://creativecommons.org/licenses/by/4.0/).

Contact
-------
If you need assistance, please post a question on the Open Data
Cube Slack channel (http://slack.opendatacube.org/) or on the GIS Stack
Exchange (https://gis.stackexchange.com/questions/ask?tags=open-data-cube)
using the `open-data-cube` tag (you can view previously asked questions
here: https://gis.stackexchange.com/questions/tagged/open-data-cube).

If you would like to report an issue with this script, you can file one on
Github: https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues/new
