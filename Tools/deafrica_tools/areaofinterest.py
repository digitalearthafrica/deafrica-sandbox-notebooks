"""
Function for defining an area of interest using either a point and buffer or a vector file.
"""

import geopandas as gpd
from geojson import Feature, FeatureCollection, Point
from shapely.geometry import box


def define_area(
    lat: float = None,
    lon: float = None,
    buffer: float = None,
    lat_buffer: float = None,
    lon_buffer: float = None,
    vector_path: str = None,
) -> FeatureCollection:
    """
    Define an area of interest using either a point and buffer or separate latitude and longitude buffers, or a vector.

    Parameters:
    -----------
    lat : float, optional
        The latitude of the center point of the area of interest.
    lon : float, optional
        The longitude of the center point of the area of interest.
    buffer : float, optional
        The buffer around the center point, in degrees. This is used if separate latitude and longitude buffers are not provided.
    lat_buffer : float, optional
        The buffer around the center point, extending along the latitude, in degrees.
    lon_buffer : float, optional
        The buffer around the center point, extending along the longitude, in degrees.
    vector_path : str, optional
        The path to a vector defining the area of interest.

    Returns:
    --------
    feature_collection : FeatureCollection
        A GeoJSON feature collection representing the area of interest.
    """
    # Check if both buffer and separate lat/lon buffers are specified
    if buffer is not None and (lat_buffer is not None or lon_buffer is not None):
        raise ValueError("Specify either buffer or separate lat_buffer and lon_buffer, not both.")

    # Check if either lat_buffer or lon_buffer is provided without the other
    if (lat_buffer is not None and lon_buffer is None) or (
        lat_buffer is None and lon_buffer is not None
    ):
        raise ValueError("Both lat_buffer and lon_buffer must be provided together.")

    # Ensure buffer values are positive
    # if negative values are provided for buffer, lat_buffer, or lon_buffer,
    # they will be converted to their absolute values without raising an error.
    if buffer is not None:
        buffer = abs(buffer)
    if lat_buffer is not None:
        lat_buffer = abs(lat_buffer)
    if lon_buffer is not None:
        lon_buffer = abs(lon_buffer)

    # Define area using point and buffer
    if lat is not None and lon is not None:
        if buffer is not None and (lat_buffer is None or lon_buffer is None):
            lat_buffer = lon_buffer = buffer

        if lat_buffer is not None and lon_buffer is not None:
            lat_range = (lat - lat_buffer, lat + lat_buffer)
            lon_range = (lon - lon_buffer, lon + lon_buffer)
            box_geom = box(min(lon_range), min(lat_range), max(lon_range), max(lat_range))
            aoi = gpd.GeoDataFrame(geometry=[box_geom], crs="EPSG:4326")
        else:
            aoi = gpd.GeoDataFrame(geometry=[Point(lon, lat).buffer(buffer)], crs="EPSG:4326")

    # Define area using vector
    elif vector_path is not None:
        aoi = gpd.read_file(vector_path).to_crs("EPSG:4326")
    # If neither option is provided, raise an error
    else:
        raise ValueError("Either lat/lon/buffer or vector_path must be provided.")

    # Convert the GeoDataFrame to a GeoJSON FeatureCollection
    features = [
        Feature(geometry=row["geometry"], properties=row.drop("geometry").to_dict())
        for _, row in aoi.iterrows()
    ]
    feature_collection = FeatureCollection(features)

    return feature_collection
