"""
Coastal analyses on Digital Earth Africa data.
"""

import geopandas as gpd
import pandas as pd
import requests
from owslib.wfs import WebFeatureService
from pandas.plotting import register_matplotlib_converters
from shapely.geometry import box

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


def tidal_tag(*args, **kwargs):
    raise ImportError(
        "The `tidal_tag` function has been removed and is no longer available in this package.\n"
        "Please install and use the `eo-tides` package instead:\n"
        "https://geoscienceaustralia.github.io/eo-tides/migration/"
    )


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
        assert (
            transects_gdf.crs == lines_gdf.crs
        ), "Please ensure both input datasets use the same CRS."

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
