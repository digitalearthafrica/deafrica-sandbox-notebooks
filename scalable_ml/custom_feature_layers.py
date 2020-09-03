
import richdem as rd
import pyproj
import dask
import hdstats
import datacube
import numpy as np
import sys
import xarray as xr
import warnings
import dask.array as da
from odc.algo import xr_reproject
from datacube.utils.geometry import assign_crs
from odc.algo import randomize, reshape_for_geomedian

sys.path.append('../Scripts')
from deafrica_bandindices import calculate_indices
from deafrica_temporal_statistics import xr_phenology, temporal_statistics
from deafrica_classificationtools import HiddenPrints
from deafrica_datahandling import load_ard
from xr_geomedian_tmad import xr_geomedian_tmad

warnings.filterwarnings("ignore")

def xr_terrain(da, attribute=None):
    """
    Using the richdem package, calculates terrain attributes
    on a DEM stored in memory as an xarray.DataArray 
    
    Params
    -------
    da : xr.DataArray
    attribute : str
        One of the terrain attributes that richdem.TerrainAttribute()
        has implemented. e.g. 'slope_riserun', 'slope_percentage', 'aspect'.
        See all option here:  
        https://richdem.readthedocs.io/en/latest/python_api.html#richdem.TerrainAttribute
        
    """
    #remove time if its there
    da = da.squeeze()
    #convert to richdem array
    rda = rd.rdarray(da.data, no_data=da.attrs['nodata'])
    #add projection and geotransform
    rda.projection=pyproj.crs.CRS(da.attrs['crs']).to_wkt()
    rda.geotransform = da.geobox.affine.to_gdal()
    #calulate attribute
    attrs = rd.TerrainAttribute(rda, attrib=attribute)

    #return as xarray DataArray
    return xr.DataArray(attrs,
                        attrs=da.attrs,
                        coords={'x':da.x, 'y':da.y},
                        dims=['y', 'x'])


def phenology_features(ds):
    dc = datacube.Datacube(app='training')
    data = calculate_indices(ds,
                             index=['NDVI'],
                             drop=True,
                             collection='s2')
    
    #temporal stats
    ts = temporal_statistics(data.NDVI,
                       stats=['f_mean', 'abs_change','discordance'
                              'complexity','central_diff'])
    
    #rainfall climatology
    print('rainfall...')
    chirps = assign_crs(xr.open_rasterio('data/CHIRPS/CHPclim_sum.nc'),  crs='epsg:4326')
    chirps = xr_reproject(chirps,ds.geobox,"mode")
    chirps = chirps.to_dataset(name='chirps')
    #chirps = chirps.mean(['x','y'])
    
    #slope
    print('slope...')
    slope = dc.load(product='srtm', like=ds.geobox).squeeze()
    slope = slope.elevation
    slope = xr_terrain(slope, 'slope_riserun')
    slope = slope.to_dataset(name='slope')
    #slope = slope.mean(['x','y'])
    
    #Surface reflectance results
    print("SR..")
    sr = ds.median('time')
    #sr = ds.mean(['x','y']).median('time')
    print('Merging...')
    result = xr.merge([ts, sr, chirps,slope], compat='override')
    result = assign_crs(result, crs=ds.geobox.crs)
    
    return result.squeeze()


def two_epochs_gm_mads(ds):
    dc = datacube.Datacube(app='training')
    
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12')) 
    
    def fun(ds, era):
        
        gm_mads = xr_geomedian_tmad(ds)
        gm_mads = calculate_indices(gm_mads,
                               index=['NDVI', 'LAI'],
                               drop=False,
                               collection='s2')
        
        for band in ds.data_vars:
            ds = ds.rename({band:band+era})
        
        return gm_mads
    
    epoch1_gm_mads = fun(ds1, era='_S1')
    epoch2_gm_mads = fun(ds2, era='_S2')
    
    slope = dc.load(product='srtm', like=ds.geobox).squeeze()
    slope = slope.elevation
    slope = xr_terrain(slope, 'slope_riserun')
    slope = slope.to_dataset(name='slope')
    
    result = xr.merge([epoch1_gm_mads,
                       epoch2_gm_mads,
                       slope], compat='override')
    
    print(result)
    return result.squeeze()
 
    
    