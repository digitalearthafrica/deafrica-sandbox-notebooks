
import os
from osgeo import gdal
import shutil
import numpy as np
import xarray as xr
import geopandas as gpd
from datacube import Datacube
from odc.algo import xr_reproject
from datacube.utils.cog import write_cog
from rsgislib.segmentation import segutils
from datacube.utils.geometry import assign_crs
from datacube.testutils.io import rio_slurp_xarray
from deafrica_tools.spatial import xr_rasterize
from deafrica_tools.classification import HiddenPrints

def post_processing(
    predicted: xr.Dataset,

) -> xr.DataArray:
    """
    filter prediction results with post processing filters.
    
    Simplified from production code to skip
    segmentation, probability, and mode calcs

    """
    
    dc = Datacube(app='whatever')
    
    predict=predicted.Predictions
    
    #--Post process masking---------------------------------------------------------------

    # mask with WDPA
    url_wdpa="s3://deafrica-input-datasets/protected_areas/WDPA_southern.tif"
    wdpa=rio_slurp_xarray(url_wdpa, gbox=predicted.geobox)
    wdpa = wdpa.astype(bool)
    predict = predict.where(~wdpa, 0)

    #mask with WOFS
    wofs=dc.load(product='wofs_ls_summary_annual',
                 like=predicted.geobox,
                 time=('2019'))
    wofs=wofs.frequency > 0.2 # threshold
    predict=predict.where(~wofs, 0)

    #mask steep slopes
    url_slope="https://deafrica-input-datasets.s3.af-south-1.amazonaws.com/srtm_dem/srtm_africa_slope.tif"
    slope=rio_slurp_xarray(url_slope, gbox=predicted.geobox)
    slope=slope > 50
    predict=predict.where(~slope, 0)

    #mask where the elevation is above 3600m
    elevation=dc.load(product='dem_srtm', like=predicted.geobox)
    elevation=elevation.elevation > 3600 # threshold
    predict=predict.where(~elevation.squeeze(), 0)
    
    #set dtype
    predict=predict.astype(np.int8)

    return predict
