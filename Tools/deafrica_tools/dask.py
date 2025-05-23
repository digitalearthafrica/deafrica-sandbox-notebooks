"""
Functions for simplifying the creation of a local dask cluster.
"""

import os
from importlib.util import find_spec

import dask
from aiohttp import ClientConnectionError
from datacube.utils.dask import start_local_dask
from datacube.utils.rio import configure_s3_access

_HAVE_PROXY = bool(find_spec("jupyter_server_proxy"))
_IS_AWS = "AWS_ACCESS_KEY_ID" in os.environ or "AWS_DEFAULT_REGION" in os.environ


def create_local_dask_cluster(
    spare_mem: str = "3Gb", display_client: bool = True, return_client: bool = False
):
    """
    Using the datacube utils function `start_local_dask`, generate
    a local dask cluster. Automatically detects if on AWS or NCI.

    Parameters
    ----------
    spare_mem : String, optional
        The amount of memory, in Gb, to leave for the notebook to run.
        This memory will not be used by the cluster. e.g '3Gb'
    display_client : Bool, optional
        An optional boolean indicating whether to display a summary of
        the dask client, including a link to monitor progress of the
        analysis. Set to False to hide this display.
    return_client : Bool, optional
        An optional boolean indicating whether to return the dask client
        object.

    """

    if _HAVE_PROXY:
        # Configure dashboard link to go over proxy
        prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
        dask.config.set({"distributed.dashboard.link": prefix + "proxy/{port}/status"})

    # Start up a local cluster
    client = start_local_dask(mem_safety_margin=spare_mem)

    if _IS_AWS:
        # Configure GDAL for s3 access
        configure_s3_access(aws_unsigned=True, client=client)

    # Show the dask cluster settings
    if display_client:
        from IPython.display import display

        display(client)

    # return the client as an object
    if return_client:
        return client


try:
    from dask_gateway import Gateway

    def create_dask_gateway_cluster(profile="r5_L", workers=2):
        """
        Create a cluster in our internal dask cluster.

        Parameters
        ----------
        profile : str
            Possible values are:
                - r5_L (2 cores, 15GB memory)
                - r5_XL (4 cores, 31GB memory)
                - r5_2XL (8 cores, 63GB memory)
                - r5_4XL (16 cores, 127GB memory)

        workers : int
            Number of workers in the cluster.
        """
        try:
            gateway = Gateway()

            # Close any existing clusters
            cluster_names = gateway.list_clusters()
            if len(cluster_names) > 0:
                print("Cluster(s) still running:", cluster_names)
                for n in cluster_names:
                    cluster = gateway.connect(n.name)
                    cluster.shutdown()

            options = gateway.cluster_options()
            options["profile"] = profile

            # limit username to alphanumeric characters
            # kubernetes pods won't launch if labels contain anything other than [a-Z, -, _]
            options["jupyterhub_user"] = "".join(
                c if c.isalnum() else "-" for c in os.getenv("JUPYTERHUB_USER")
            )

            cluster = gateway.new_cluster(options)
            cluster.scale(workers)
            return cluster
        except ClientConnectionError:
            raise ConnectionError("access to dask gateway cluster unauthorized")

except ImportError:

    def create_dask_gateway_cluster(*args, **kwargs):
        raise NotImplementedError
