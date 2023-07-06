"""
Function for defining an area of interest using either a point and buffer or a shapefile file. 
"""

# Import required packages

# Force GeoPandas to use Shapely instead of PyGEOS
# In a future release, GeoPandas will switch to using Shapely by default.
import os
os.environ['USE_PYGEOS'] = '0'

import geopandas as gpd
from shapely.geometry import box
from geojson import Feature, Point, FeatureCollection

def define_area(lat=None, lon=None, buffer=None, shapefile_path=None):
    '''
    Define an area of interest using either a point and buffer or a shapefile.
    
    Parameters:
    -----------
    lat : float, optional
        The latitude of the center point of the area of interest.
    lon : float, optional
        The longitude of the center point of the area of interest.
    buffer : float, optional
        The buffer around the center point, in degrees.
    shapefile_path : str, optional
        The path to a shapefile defining the area of interest.
    
    Returns:
    --------
    feature_collection : dict
        A GeoJSON feature collection representing the area of interest.
    '''
    # Define area using point and buffer
    if lat is not None and lon is not None and buffer is not None:
        lat_range = (lat - buffer, lat + buffer)
        lon_range = (lon - buffer, lon + buffer)
        box_geom = box(min(lon_range), min(lat_range), max(lon_range), max(lat_range))
        aoi = gpd.GeoDataFrame(geometry=[box_geom], crs='EPSG:4326')
    
    # Define area using shapefile
    elif shapefile_path is not None:
        aoi = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    # If neither option is provided, raise an error
    else:
        raise ValueError("Either lat/lon/buffer or shapefile_path must be provided.")
    
    # Convert the GeoDataFrame to a GeoJSON FeatureCollection
    features = [Feature(geometry=row["geometry"], properties=row.drop("geometry").to_dict()) for _, row in aoi.iterrows()]
    feature_collection = FeatureCollection(features)
    
    return feature_collection