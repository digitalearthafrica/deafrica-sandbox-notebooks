"""
Functions for simplifying the creation of a local dask cluster.

License
-------
The code in this notebook is licensed under the Apache License,
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Digital Earth
Africa data is licensed under the Creative Commons by Attribution 4.0
license (https://creativecommons.org/licenses/by/4.0/).

Contact
-------
If you need assistance, please post a question on the Open Data
Cube Slack channel (http://slack.opendatacube.org/) or on the GIS Stack
Exchange (https://gis.stackexchange.com/questions/ask?tags=open-data-cube)
using the `open-data-cube` tag (you can view previously asked questions
here: https://gis.stackexchange.com/questions/tagged/open-data-cube).

If you would like to report an issue with this script, you can file one on
Github https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues


.. autosummary::
   :nosignatures:
   :toctree: gen

"""

import os
import dask
from datacube.utils.dask import start_local_dask
from datacube.utils.rio import configure_s3_access
from aiohttp import ClientConnectionError


def create_local_dask_cluster(
    spare_mem="3Gb", aws_unsigned=True, display_client=True, **kwargs
):
    """
    Using the datacube utils function 'start_local_dask', generate
    a local dask cluster.

    Example use :

        from deafrica_tools.dask import create_local_dask_cluster

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
    dask.config.set(
        {
            "distributed.dashboard.link": os.environ.get(
                "JUPYTERHUB_SERVICE_PREFIX", "/"
            )
            + "proxy/{port}/status"
        }
    )

    # start up a local cluster
    client = start_local_dask(mem_safety_margin=spare_mem, **kwargs)

    ## Configure GDAL for s3 access
    configure_s3_access(aws_unsigned=aws_unsigned, client=client)

    # Show the dask cluster settings
    if display_client:
        display(client)

try:
    from dask_gateway import Gateway

    def create_dask_gateway_cluster(profile='r5_XL', workers=2):
        """
        Create a cluster in our internal dask cluster.

        Parameters
        ----------
        profile : str
            Possible values are: XL (2 cores, 15GB memory), 2XL (4 cores, 31GB memory), 4XL (8 cores, 62GB memory)
        workers : int
            Number of workers in the cluster.

        """
        try:
            gateway = Gateway()

            options = gateway.cluster_options()
            options['profile'] = profile
            ## This Configuration is used for dask-worker pod labels
            options['jupyterhub_user'] = os.getenv('JUPYTERHUB_USER')

            cluster = gateway.new_cluster(options)
            cluster.scale(workers)
            return cluster

        except ClientConnectionError:
            raise ConnectionError("access to dask gateway cluster unauthorized")


except ImportError:
    def create_dask_gateway_cluster(*args, **kwargs):
        raise NotImplementedError
