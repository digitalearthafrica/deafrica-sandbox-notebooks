"""
Functions to retrieve iSDAsoil data.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio as rio
from pyproj import Transformer
import matplotlib.pyplot as plt
import os
import numpy as np

from urllib.parse import urlparse
import boto3
from pystac import stac_io, Catalog

#this function allows us to directly query the data on s3, adapted from iSDA tutorial https://github.com/iSDA-Africa/isdasoil-tutorial/blob/main/iSDAsoil-tutorial.ipynb
def my_read_method(uri):
    parsed = urlparse(uri)
    if parsed.scheme == 's3':
        bucket = parsed.netloc
        key = parsed.path[1:]
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, key)
        return obj.get()['Body'].read().decode('utf-8')
    else:
        return stac_io.default_read_text_method(uri)

stac_io.read_text_method = my_read_method

catalog = Catalog.from_file("https://isdasoil.s3.amazonaws.com/catalog.json")

assets = {}

for root, catalogs, items in catalog.walk():
    for item in items:
        str(f"Type: {item.get_parent().title}")
        # save all items to a dictionary as we go along
        assets[item.id] = item
        for asset in item.assets.values():
            if asset.roles == ['data']:
                str(f"Title: {asset.title}")
                str(f"Description: {asset.description}")
                str(f"URL: {asset.href}")
                str("------------")

# define load_isda() function

def load_isda(var, lat, lon):
    """
    Download and return iSDA variable with number of bands corresponding to number of iSDA layers.
    Parameters
    ----------
    var : string
        Name of the iSDA variable to download, e.g "ph"
    lat: tuple or list
        Latitude range for query.
    lon: tuple or list
        Longitude range for query. 
    """

    bands = assets[var].assets["image"].extra_fields.get('eo:bands')
    bands = [val['description'] for val in bands]
    
    if len(np.unique(bands)) > 1:
    
        ds = xr.open_dataset(assets[var].assets["image"].href, engine="rasterio").rio.clip_box(
                minx=lon[0],
                miny=lat[0],
                maxx=lon[1],
                maxy=lat[1],
                crs="EPSG:4326",
            )

        ds_layered = ds.drop_dims('band')
        for x in np.unique(ds.band):
            ds_layered[bands[x-1]] = ds.sel(band=x).to_array(dim='band').squeeze()
            
    else:
        
        ds_layered = xr.open_dataset(assets[var].assets["image"].href, engine="rasterio").rio.clip_box(
                minx=lon[0],
                miny=lat[0],
                maxx=lon[1],
                maxy=lat[1],
                crs="EPSG:4326",
            ).squeeze()

    return ds_layered
