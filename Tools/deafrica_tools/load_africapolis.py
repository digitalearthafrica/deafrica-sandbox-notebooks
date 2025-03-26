"""
Loading OECD SWAC Africapolis data.
Last modified: December 2024
"""

import requests
import geopandas as gpd
from shapely.geometry import box
from owslib.wfs import WebFeatureService


WFS_ADDRESS = "https://geoserver.digitalearth.africa/geoserver/wfs"

def get_africapolis(
    bbox, crs="EPSG:4326", layer=None
) -> gpd.GeoDataFrame:
    """
    Retrieve Africapolis data from a GeoServer WFS for a given bounding box.
    Parameters
    ----------
    bbox : tuple or GeoDataFrame
        Bounding box as (xmin, ymin, xmax, ymax) in the specified CRS,
        or a GeoPandas object to extract bounds from.
    crs : str
        Coordinate Reference System for the bounding box. Defaults to EPSG:4326.
    layer : str
        Africapolis layer to load ('africapolis_2015' or 'africapolis_2020'). 
    drop_wms : bool
        Drop visualization-specific columns. Defaults to True.
    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing Africapolis data.
    """
    # Extract bounding box if a GeoDataFrame is provided
    if isinstance(bbox, gpd.GeoDataFrame) or isinstance(bbox, gpd.GeoSeries):
        crs = str(bbox.crs)
        bbox = bbox.total_bounds
    # Validate the layer choice (either 2015 or 2020)
    describe_url = (
        f"https://geoserver.digitalearth.africa/geoserver/wms?service=WMS&version=1.1.1&"
        f"request=DescribeLayer&layers=africapolis:{layer}&outputFormat=application/json"
    )
    try:
        describe_response = requests.get(describe_url).json()
        available_layers = [
            l["layerName"] for l in describe_response.get("layerDescriptions", [])
        ]
        # if layer not in available_layers:
        #     raise ValueError(f"Layer {layer} is not available. Available layers: {available_layers}")
    except Exception as e:
        raise RuntimeError(f"Error retrieving available layers: {e}")
    # Query the WFS for the specified layer and bbox
    try:
        wfs = WebFeatureService(url=WFS_ADDRESS, version="2.0.0")
        response = wfs.getfeature(
            typename=f"africapolis:{layer}",
            bbox=tuple(bbox) + (crs,),
            outputFormat="application/json"
        )
    except Exception as e:
        raise RuntimeError(f"Error querying WFS: {e}")
    # Load data into a GeoDataFrame
    africapolis_gdf = gpd.read_file(response)

    return africapolis_gdf