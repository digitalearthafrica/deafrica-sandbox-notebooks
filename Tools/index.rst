DE Africa Tools Package
=======================

``deafrica_tools`` is a Python package containing modules for loading,
analysing, and exporting Digital Earth Africa data. It is automatically
installed in the Digital Earth Africa Sandbox environment.

For installation instructions, see the `Tools directory
<https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/tree/master/Tools/>`_
in the GitHub repository.

Core modules
------------

.. autosummary::
   :toctree: gen

   deafrica_tools.areaofinterest
   deafrica_tools.bandindices
   deafrica_tools.classification
   deafrica_tools.coastal
   deafrica_tools.dask
   deafrica_tools.datahandling
   deafrica_tools.externaldrive
   deafrica_tools.load_africapolis
   deafrica_tools.load_era5
   deafrica_tools.load_isda
   deafrica_tools.load_soil_moisture
   deafrica_tools.load_wapor
   deafrica_tools.methane_convention
   deafrica_tools.plotting
   deafrica_tools.spatial
   deafrica_tools.temporal
   deafrica_tools.waterbodies
   deafrica_tools.wetlands

Apps and widgets
----------------

Applications and widgets are available through ``deafrica_tools.app``.

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

The code in this package is licensed under the `Apache License, Version 2.0
<https://www.apache.org/licenses/LICENSE-2.0>`_.

Digital Earth Africa data is licensed under the `Creative Commons
Attribution 4.0 International License
<https://creativecommons.org/licenses/by/4.0/>`_.

Contact
-------

For assistance, post a question in the `Digital Earth Africa Slack workspace
<https://join.slack.com/t/digitalearthafrica/shared_invite/zt-4cu3w53g0-IpKwdWM4zOzgg0ijQXbRHg/>`_ or on `GIS Stack Exchange
<https://gis.stackexchange.com/questions/ask?tags=open-data-cube>`_
using the ``open-data-cube`` tag.

You can also browse `previous questions tagged open-data-cube
<https://gis.stackexchange.com/questions/tagged/open-data-cube>`_.

To report an issue with this package, open an issue in the `GitHub repository
<https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues/new>`_.