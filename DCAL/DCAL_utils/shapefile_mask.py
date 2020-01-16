import fiona
import xarray as xr
import numpy as np

from rasterio.features import geometry_mask
import shapely
from shapely.ops import transform
from shapely.geometry import shape
from functools import partial
import pyproj


def shapefile_mask(dataset: xr.Dataset, shapefile) -> np.array:
    """Extracts a mask from a shapefile using dataset latitude and longitude extents.

    Args:
        dataset (xarray.Dataset): The dataset with the latitude and longitude extents.
        shapefile (string): The shapefile to be used for extraction.

    Returns:
        A boolean mask array.
    """
    with fiona.open(shapefile, 'r') as source:
        collection = list(source)
        geometries = []
        for feature in collection:
            geom = shape(feature['geometry'])
            project = partial(
                pyproj.transform,
                pyproj.Proj(init=source.crs['init']), # source crs
                pyproj.Proj(init='epsg:4326')) # destination crs
            geom = transform(project, geom)  # apply projection
            geometries.append(geom)
        geobox = dataset.geobox
        mask = geometry_mask(
            geometries,
            out_shape=geobox.shape,
            transform=geobox.affine,
            all_touched=True,
            invert=True)
    return mask