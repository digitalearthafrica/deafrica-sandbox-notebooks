## deafrica_dask.py
'''
Description: A set of python functions for simplifying the creation of a 
local dask cluster.

License: The code in this notebook is licensed under the Apache License,
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Digital Earth 
Africa data is licensed under the Creative Commons by Attribution 4.0 
license (https://creativecommons.org/licenses/by/4.0/).

Contact: If you need assistance, please post a question on the Open Data 
Cube Slack channel (http://slack.opendatacube.org/) or on the GIS Stack 
Exchange (https://gis.stackexchange.com/questions/ask?tags=open-data-cube) 
using the `open-data-cube` tag (you can view previously asked questions 
here: https://gis.stackexchange.com/questions/tagged/open-data-cube). 

If you would like to report an issue with this script, you can file one on 
Github https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues

Functions included:
    create_local_dask_cluster
    
Last modified: March 2020

'''

import os
import dask
from datacube.utils.dask import start_local_dask
from datacube.utils.rio import configure_s3_access


def create_local_dask_cluster(spare_mem='3Gb',
                              aws_unsigned= True,
                              display_client=True,
                              **kwargs):
    """
    Using the datacube utils function 'start_local_dask', generate
    a local dask cluster.
    
    Example use :
        
        import sys
        sys.path.append("../Scripts")
        from deafrica_dask import create_local_dask_cluster
        
        create_local_dask_cluster(spare_mem='4Gb')
    
    Parameters
    ----------  
    spare_mem : String, optional
        The amount of memory, in Gb, to leave for the notebook to run.
        This memory will not be used by the cluster. e.g '3Gb'
    aws_unsigned : Bool, optional
         This parameter determines if credentials for S3 access are required and
         passes them on to processing threads, either local or on dask cluster. 
         Set to True if working with publicly available datasets, and False if
         working with private data. i.e if loading Landsat C2 provisional data set 
         this to aws_unsigned=False
    display_client : Bool, optional
        An optional boolean indicating whether to display a summary of
        the dask client, including a link to monitor progress of the
        analysis. Set to False to hide this display.
    **kwargs:
        Additional keyword arguments that will be passed to start_local_dask().
        E.g. n_workers can be set to be greater than 1.
    """

    # configure dashboard link to go over proxy
    dask.config.set({"distributed.dashboard.link":
                 os.environ.get('JUPYTERHUB_SERVICE_PREFIX', '/')+"proxy/{port}/status"})

    # start up a local cluster  
    client = start_local_dask(mem_safety_margin=spare_mem, **kwargs)

    ## Configure GDAL for s3 access
    configure_s3_access(aws_unsigned=aws_unsigned,  
                        client=client);

    # Show the dask cluster settings
    if display_client:
        display(client)
