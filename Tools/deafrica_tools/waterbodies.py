"""
Loading and processing DE Africa Water Bodies data.
Last modified: November 2023
"""

from datetime import datetime

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from owslib.etree import etree
from owslib.fes import PropertyIsEqualTo
from owslib.wfs import WebFeatureService

# URL for the DE Africa Water Bodies data on PROD Geoserver.
WFS_ADDRESS = "https://geoserver.digitalearth.africa/geoserver/wfs"
WFS_LAYER = "waterbodies:DEAfrica_Waterbodies"
API_ADDRESS = "https://api.digitalearth.africa/waterbodies/"


def get_waterbody(geohash: str) -> gpd.GeoDataFrame:
    """Gets a waterbody polygon and metadata by geohash.

    Parameters
    ----------
    geohash : str
        The geohash/uid for a waterbody in DE Africa Water Bodies.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the polygon.
    """

    wfs = WebFeatureService(url=WFS_ADDRESS, version="1.1.0")
    filter_ = PropertyIsEqualTo(propertyname="uid", literal=geohash)
    filterxml = etree.tostring(filter_.toXML()).decode("utf-8")
    response = wfs.getfeature(
        typename=WFS_LAYER,
        filter=filterxml,
        outputFormat="json",
    )
    wb_gpd = gpd.read_file(response)
    return wb_gpd


def get_waterbodies(bbox: tuple, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Gets the polygons and metadata for multiple water bodies by bbox.

    Parameters
    ----------
    bbox : (xmin, ymin, xmax, ymax)
        Bounding box.
    crs : str
        Optional CRS for the bounding box.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the polygons and metadata.
    """

    wfs = WebFeatureService(url=WFS_ADDRESS, version="1.1.0")
    response = wfs.getfeature(
        typename=WFS_LAYER,
        bbox=tuple(bbox) + (crs,),
        outputFormat="json",
    )
    wb_gpd = gpd.read_file(response)
    return wb_gpd


def get_geohashes(bbox: tuple = None, crs: str = "EPSG:4326") -> list[str]:
    """Gets all waterbody geohashes.

    Parameters
    ----------
    bbox : (xmin, ymin, xmax, ymax)
        Optional bounding box.
    crs : str
        Optional CRS for the bounding box.

    Returns
    -------
    [str]
        A list of geohashes.
    """

    wfs = WebFeatureService(url=WFS_ADDRESS, version="1.1.0")
    if bbox is not None:
        bbox = bbox + (crs,)
    response = wfs.getfeature(
        typename=WFS_LAYER,
        propertyname="uid",
        outputFormat="json",
        bbox=bbox,
    )
    wb_gpd = gpd.read_file(response)
    return list(wb_gpd["uid"])


def get_time_series(
    geohash: str = None,
    waterbody: pd.Series = None,
    start_date: str = "1984-01-01",
    end_date: str = datetime.now().strftime("%Y-%m-%d"),
) -> pd.DataFrame:
    """Gets the time series for a waterbody. Specify either a GeoDataFrame row or a geohash.

    Parameters
    ----------
    geohash : str
        The geohash/uid for a waterbody in DE Africa Water Bodies.
    waterbody : pd.Series
        One row of a GeoDataFrame representing a waterbody.
    start_date : str
        Start date for the time range to filter the timeseries to.
    end_date : str
        End date for the time range to filter the timeseries to.
    Returns
    -------
    pd.DataFrame
        A time series for the waterbody.
    """
    if waterbody is not None and geohash is not None:
        raise ValueError("One of waterbody and geohash must be None")
    if waterbody is None and geohash is None:
        raise ValueError("One of waterbody and geohash must be specified")

    if geohash is not None:
        wb = get_waterbody(geohash)
        wb_id = wb.wb_id.item()
    else:
        wb_id = waterbody.wb_id.item()
    url = (
        API_ADDRESS
        + f"waterbody/{wb_id}/observations/csv?start_date={start_date}&end_date={end_date}"
    )
    wb_timeseries = pd.read_csv(url)
    # Tidy up the dataframe.
    wb_timeseries = wb_timeseries.set_index("date")
    wb_timeseries.index = pd.to_datetime(wb_timeseries.index)
    # Create a rolling median for the wet time series
    wb_timeseries["percent_wet_rolling_median"] = wb_timeseries["percent_wet"].rolling(3).median()

    return wb_timeseries


def display_time_series(wb_timeseries: pd.DataFrame = None) -> None:
    """Displays the timeseries as an interactive plot

    Parameters
    ----------
    wb_timeseries : pd.DataFrame
        A time series for the waterbody.

    Returns
    -------
    None
    """

    fig = go.Figure()

    # Add a scatter plot of invalid percentage measruements
    fig.add_trace(
        go.Scatter(
            x=wb_timeseries.index,
            y=wb_timeseries["percent_invalid"],
            mode="markers",
            marker=dict(color="red"),
            name="Invalid Percentage",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Invalid: %{y:.2f}%<extra></extra>",
            opacity=0.7,
        )
    )

    # Add a line for the median wet percentage
    fig.add_trace(
        go.Scatter(
            x=wb_timeseries.index,
            y=wb_timeseries["percent_wet_rolling_median"],
            mode="lines",
            line=dict(color="blue"),
            opacity=0.3,
            name="Wet Percentage - Rolling Median",
        )
    )

    # Add a scatter plot of the wet percentage measurements
    fig.add_trace(
        go.Scatter(
            x=wb_timeseries.index,
            y=wb_timeseries["percent_wet"],
            mode="markers",
            marker=dict(color="blue"),
            name="Wet Percentage",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Wet: %{y:.2f}%<extra></extra>",
            opacity=0.7,
        )
    )

    # Customize layout
    fig.update_layout(
        title="Wet surface area time series",
        xaxis_title="Date",
        yaxis_title="Percentage",
        yaxis_range=[-5, 105],
    )

    # Display the plot
    fig.show()
