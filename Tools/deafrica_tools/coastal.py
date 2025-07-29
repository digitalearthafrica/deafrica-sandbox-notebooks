"""
Coastal analyses on Digital Earth Africa data.
"""

import os
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import odc.algo
import odc.geo.xr
import pandas as pd
import pyproj
import pyTMD.io
import pyTMD.utilities
import requests
import timescale
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject
from owslib.wfs import WebFeatureService
from pandas.plotting import register_matplotlib_converters
from scipy import stats
from shapely.geometry import box

from deafrica_tools.datahandling import parallel_apply

# Fix converters for tidal plot
register_matplotlib_converters()


# URL for the DE Africa Coastlines data on Geoserver.
WFS_ADDRESS = "https://geoserver.digitalearth.africa/geoserver/wfs"


def model_tides(*args, **kwargs):
    raise ImportError(
        "The `model_tides` function has been removed and is no longer available in this package.\n"
        "Please install and use the `eo-tides` package instead:\n"
        "https://geoscienceaustralia.github.io/eo-tides/migration/"
    )


def pixel_tides(*args, **kwargs):
    raise ImportError(
        "The `pixel_tides` function has been removed and is no longer available in this package.\n"
        "Please install and use the `eo-tides` package instead:\n"
        "https://geoscienceaustralia.github.io/eo-tides/migration/"
    )



def tidal_tag(
    ds,
    ebb_flow=False,
    swap_dims=False,
    tidepost_lat=None,
    tidepost_lon=None,
    return_tideposts=False,
    **model_tides_kwargs,
):
    """
    Takes an xarray.Dataset and returns the same dataset with a new
    `tide_m` variable giving the height of the tide at the exact
    moment of each satellite acquisition.

    The function models tides at the centroid of the dataset by default,
    but a custom tidal modelling location can be specified using
    `tidepost_lat` and `tidepost_lon`.

    The default settings use the FES2014 global tidal model, implemented
    using the pyTMD Python package. FES2014 was produced by NOVELTIS,
    LEGOS, CLS Space Oceanography Division and CNES. It is distributed
    by AVISO, with support from CNES (http://www.aviso.altimetry.fr/).

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray.Dataset object with x, y and time dimensions
    ebb_flow : bool, optional
        An optional boolean indicating whether to compute if the
        tide phase was ebbing (falling) or flowing (rising) for each
        observation. The default is False; if set to True, a new
        `ebb_flow` variable will be added to the dataset with each
        observation labelled with 'Ebb' or 'Flow'.
    swap_dims : bool, optional
        An optional boolean indicating whether to swap the `time`
        dimension in the original xarray.Dataset to the new
        `tide_m` variable. Defaults to False.
    tidepost_lat, tidepost_lon : float or int, optional
        Optional coordinates used to model tides. The default is None,
        which uses the centroid of the dataset as the tide modelling
        location.
    return_tideposts : bool, optional
        An optional boolean indicating whether to return the `tidepost_lat`
        and `tidepost_lon` location used to model tides in addition to the
        xarray.Dataset. Defaults to False.
    **model_tides_kwargs :
        Optional parameters passed to the `eo_tides.model.model_tides`
        function. Important parameters include "model" and "directory",
        used to specify the tide model to use and the location of its files.

    Returns
    -------
    The original xarray.Dataset with a new `tide_m` variable giving
    the height of the tide (and optionally, its ebb-flow phase) at the
    exact moment of each satellite acquisition (if `return_tideposts=True`,
    the function will also return the `tidepost_lon` and `tidepost_lat`
    location used in the analysis).

    """
    from eo_tides.model import model_tides
    # If custom tide modelling locations are not provided, use the
    # dataset centroid
    if not tidepost_lat or not tidepost_lon:

        tidepost_lon, tidepost_lat = ds.odc.geobox.geographic_extent.centroid.coords[0]
        print(
            f"Setting tide modelling location from dataset centroid: "
            f"{tidepost_lon:.2f}, {tidepost_lat:.2f}"
        )

    else:
        print(
            f"Using user-supplied tide modelling location: "
            f"{tidepost_lon:.2f}, {tidepost_lat:.2f}"
        )

    # Use tidal model to compute tide heights for each observation:
    if "model" not in model_tides_kwargs:
        model_tides_kwargs["model"] = "FES2014"
    if "directory" not in model_tides_kwargs:
        model_tides_kwargs["directory"] = "/var/share/tide_models"
        
    model = model_tides_kwargs["model"]
    print(f"Modelling tides using {model} tidal model")
    tide_df = model_tides(
        x=tidepost_lon,
        y=tidepost_lat,
        time=ds.time,
        crs="EPSG:4326",
        **model_tides_kwargs,
    )

    # If tides cannot be successfully modeled (e.g. if the centre of the
    # xarray dataset is located is over land), raise an exception
    if tide_df.tide_m.isnull().all():

        raise ValueError(
            f"Tides could not be modelled for dataset centroid located "
            f"at {tidepost_lon:.2f}, {tidepost_lat:.2f}. This can occur if "
            f"this coordinate occurs over land. Please manually specify "
            f"a tide modelling location located over water using the "
            f"`tidepost_lat` and `tidepost_lon` parameters."
        )

    # Assign tide heights to the dataset as a new variable
    ds["tide_m"] = xr.DataArray(tide_df.tide_m, coords=[ds.time])

    # Optionally calculate the tide phase for each observation
    if ebb_flow:

        # Model tides for a time 15 minutes prior to each previously
        # modelled satellite acquisition time. This allows us to compare
        # tide heights to see if they are rising or falling.
        print("Modelling tidal phase (e.g. ebb or flow)")
        tide_pre_df = model_tides(
            x=tidepost_lon,
            y=tidepost_lat,
            time=(ds.time - pd.Timedelta("15 min")),
            crs="EPSG:4326",
            **model_tides_kwargs,
        )

        # Compare tides computed for each timestep. If the previous tide
        # was higher than the current tide, the tide is 'ebbing'. If the
        # previous tide was lower, the tide is 'flowing'
        tidal_phase = [
            "Ebb" if i else "Flow" for i in tide_pre_df.tide_m.values > tide_df.tide_m.values
        ]

        # Assign tide phase to the dataset as a new variable
        ds["ebb_flow"] = xr.DataArray(tidal_phase, coords=[ds.time])

    # If swap_dims = True, make tide height the primary dimension
    # instead of time
    if swap_dims:

        # Swap dimensions and sort by tide height
        ds = ds.swap_dims({"time": "tide_m"})
        ds = ds.sortby("tide_m")
        ds = ds.drop_vars("time")

    if return_tideposts:
        return ds, tidepost_lon, tidepost_lat
    else:
        return ds


def tidal_stats(*args, **kwargs):
    raise ImportError(
        "The `tidal_stats` function has been removed and is no longer available in this package.\n"
        "Please install and use the `eo-tides` package instead:\n"
        "https://geoscienceaustralia.github.io/eo-tides/migration/"
    )


def transect_distances(transects_gdf, lines_gdf, mode="distance"):
    """
    Take a set of transects (e.g. shore-normal beach survey lines), and
    determine the distance along the transect to each object in a set of
    lines (e.g. shorelines). Distances are measured in the CRS of the
    input datasets.

    For coastal applications, transects should be drawn from land to
    water (with the first point being on land so that it can be used
    as a consistent location from which to measure distances.

    The distance calculation can be performed using two modes:
        - 'distance': Distances are measured from the start of the
          transect to where it intersects with each line. Any transect
          that intersects a line more than once is ignored. This mode is
          useful for measuring e.g. the distance to the shoreline over
          time from a consistent starting location.
        - 'width' Distances are measured between the first and last
          intersection between a transect and each line. Any transect
          that intersects a line only once is ignored. This is useful
          for e.g. measuring the width of a narrow area of coastline over
          time, e.g. the neck of a spit or tombolo.

    Parameters
    ----------
    transects_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing one or multiple vector profile lines.
        The GeoDataFrame's index column will be used to name the rows in
        the output distance table.
    lines_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing one or multiple vector line features
        that intersect the profile lines supplied to `transects_gdf`.
        The GeoDataFrame's index column will be used to name the columns
        in the output distance table.
    mode : string, optional
        Whether to use 'distance' (for measuring distances from the
        start of a profile) or 'width' mode (for measuring the width
        between two profile intersections). See docstring above for more
        info; defaults to 'distance'.

    Returns
    -------
    distance_df : pandas.DataFrame
        A DataFrame containing distance measurements for each profile
        line (rows) and line feature (columns).
    """

    import warnings

    from shapely.errors import ShapelyDeprecationWarning
    from shapely.geometry import Point

    def _intersect_dist(transect_gdf, lines_gdf, mode=mode):
        """
        Take an individual transect, and determine the distance along
        the transect to each object in a set of lines (e.g. shorelines).
        """

        # Identify intersections between transects and lines
        intersect_points = lines_gdf.apply(
            lambda x: x.geometry.intersection(transect_gdf.geometry), axis=1
        )

        # In distance mode, identify transects with one intersection only,
        # and use this as the end point and the start of the transect as the
        # start point when measuring distances
        if mode == "distance":
            start_point = Point(transect_gdf.geometry.coords[0])
            point_df = intersect_points.apply(
                lambda x: (
                    pd.Series({"start": start_point, "end": x})
                    if x.type == "Point"
                    else pd.Series({"start": None, "end": None})
                )
            )

        # In width mode, identify transects with multiple intersections, and
        # use the first intersection as the start point and the second
        # intersection for the end point when measuring distances
        if mode == "width":
            point_df = intersect_points.apply(
                lambda x: (
                    pd.Series({"start": x.geoms[0], "end": x.geoms[-1]})
                    if x.type == "MultiPoint"
                    else pd.Series({"start": None, "end": None})
                )
            )

        # Calculate distances between valid start and end points
        distance_df = point_df.apply(lambda x: x.start.distance(x.end) if x.start else None, axis=1)

        return distance_df

    # Run code after ignoring Shapely pre-v2.0 warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

        # Assert that both datasets use the same CRS
        assert transects_gdf.crs == lines_gdf.crs, (
            "Please ensure both " "input datasets use the same CRS."
        )

        # Run distance calculations
        distance_df = transects_gdf.apply(lambda x: _intersect_dist(x, lines_gdf), axis=1)

        return pd.DataFrame(distance_df)


def get_coastlines(
    bbox: tuple, crs="EPSG:4326", layer="shorelines", drop_wms=True
) -> gpd.GeoDataFrame:
    """
    Get DE Africa Coastlines data for a provided bounding box using WFS.

    For a full description of the DE Africa Coastlines dataset, refer to the
    official Digital Earth Africa product description:

    Parameters
    ----------
    bbox : (xmin, ymin, xmax, ymax), or geopandas object
        Bounding box expressed as a tuple. Alternatively, a bounding
        box can be automatically extracted by suppling a
        geopandas.GeoDataFrame or geopandas.GeoSeries.
    crs : str, optional
        Optional CRS for the bounding box. This is ignored if `bbox`
        is provided as a geopandas object.
    layer : str, optional
        Which DE Africa Coastlines layer to load. Options include the annual
        shoreline vectors ("shorelines") and the rates of change
        statistics points ("statistics"). Defaults to "shorelines".
    drop_wms : bool, optional
        Whether to drop WMS-specific attribute columns from the data.
        These columns are used for visualising the dataset on DE Africa Maps,
        and are unlikely to be useful for scientific analysis. Defaults
        to True.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing shoreline or point features and
        associated metadata.
    """

    # If bbox is a geopandas object, convert to bbox.
    try:
        crs = str(bbox.crs)
        bbox = bbox.total_bounds
    except Exception:
        pass

    # Get the available layers in the coastlines:DEAfrica_Coastlines group.
    describe_layer_url = (
        "https://geoserver.digitalearth.africa/geoserver/wms?service=WMS&version=1.1.1&"
        "request=DescribeLayer&layers=coastlines:DEAfrica_Coastlines&outputFormat=application/json"
    )
    describe_layer_response = requests.get(describe_layer_url).json()
    available_layers = [
        layer["layerName"] for layer in describe_layer_response["layerDescriptions"]
    ]

    # Get the layer name.
    if layer == "shorelines":
        layer_name = [i for i in available_layers if "shorelines" in i]
    else:
        layer_name = [i for i in available_layers if "rates_of_change" in i]

    # Query WFS.
    wfs = WebFeatureService(url=WFS_ADDRESS, version="1.1.0")
    response = wfs.getfeature(typename=layer_name, bbox=tuple(bbox) + (crs,), outputFormat="json")

    # Load data as a geopandas.GeoDataFrame.
    coastlines_gdf = gpd.read_file(response)

    # Clip to extent of bounding box.
    extent = gpd.GeoSeries(box(*bbox), crs=crs).to_crs(coastlines_gdf.crs)
    coastlines_gdf = coastlines_gdf.clip(extent)

    # Optionally drop WMS-specific columns.
    if drop_wms:
        coastlines_gdf = coastlines_gdf.loc[:, ~coastlines_gdf.columns.str.contains("wms_")]

    return coastlines_gdf
