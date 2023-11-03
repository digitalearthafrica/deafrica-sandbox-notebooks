"""
Loading and processing DE Africa Water Bodies data.
Last modified: November 2023
"""

# Import required packages
import geopandas as gpd
from owslib.wfs import WebFeatureService
from owslib.fes import PropertyIsEqualTo
from owslib.etree import etree
import pandas as pd

# URL for the DE Africa Water Bodies data on Dev Geoserver.
WFS_ADDRESS = "https://geoserver.digitalearth.africa/geoserver/wfs"
WFS_LAYER = "waterbodies:DEAfrica_Waterbodies"

def get_waterbody(geohash: str) -> gpd.GeoDataFrame:
    """Gets a waterbody polygon and metadata by geohash.
    
    Parameters
    ----------
    geohash : str
        The geohash/UID for a waterbody in DE Africa Water Bodies.
    
    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the polygon.
    """
    
    wfs = WebFeatureService(url=WFS_ADDRESS, version="1.1.0")
    filter_ = PropertyIsEqualTo(propertyname="UID", literal=geohash)
    filterxml = etree.tostring(filter_.toXML()).decode("utf-8")
    response = wfs.getfeature(
        typename=WFS_LAYER,
        filter=filterxml,
        outputFormat="json",
    )
    wb_gpd = gpd.read_file(response)
    return wb_gpd


def get_waterbodies(bbox: tuple, crs="EPSG:4326") -> gpd.GeoDataFrame:
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


def get_geohashes(bbox: tuple = None, crs: str = "EPSG:4326") -> [str]:
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
        propertyname="UID",
        outputFormat="json",
        bbox=bbox,
    )
    wb_gpd = gpd.read_file(response)
    return list(wb_gpd["UID"])


def get_time_series(geohash: str = None, waterbody: pd.Series = None) -> pd.DataFrame:
    """Gets the time series for a waterbody. Specify either a GeoDataFrame row or a geohash.
    
    Parameters
    ----------
    geohash : str
        The geohash/UID for a waterbody in DE Africa Water Bodies.
    waterbody : pd.Series
        One row of a GeoDataFrame representing a waterbody.
    
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
        url = wb.timeseries[0]
    else:
        url = waterbody.timeseries
    wb_timeseries = pd.read_csv(url)
    # Tidy up the dataframe.
    wb_timeseries.dropna(inplace=True)
    wb_timeseries.columns = ["date", "pc_wet", "px_wet", "area_wet_m2", "pc_dry", "px_dry", "area_dry_m2", "pc_invalid", "px_invalid", "area_invalid_m2"]
    wb_timeseries = wb_timeseries.set_index("date")
    wb_timeseries.index = pd.to_datetime(wb_timeseries.index)
    return wb_timeseries
