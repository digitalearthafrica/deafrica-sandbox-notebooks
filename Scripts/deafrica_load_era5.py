## deafrica_load_era5.py
'''
Description: A set of python functions to retrieve ERA5 gridded climiate data.

Adpated from scripts by Andrew Cherry and Brian Killough.

License: The code in this notebook is licensed under the Apache License,
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Digital Earth 
Africa data is licensed under the Creative Commons by Attribution 4.0 
license (https://creativecommons.org/licenses/by/4.0/).

Contact: If you need assistance, please post a question on the Open Data 
Cube Slack channel (http://slack.opendatacube.org/) or on the GIS Stack 
Exchange (https://gis.stackexchange.com/questions/ask?tags=open-data-cube) 
using the `open-data-cube` tag (you can view previously asked questions 
here: https://gis.stackexchange.com/questions/tagged/open-data-cube). 

If you would like to report an issue with this script, you can file one on 
Github https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues

Functions included:
    get_era5_daily
    era5_area_crop
    era5_area_nearest
    load_era5
    
    
Last modified: April 2020

'''

import os, datetime
import numpy as np
from dateutil.parser import parse
import boto3, botocore
import xarray as xr
import warnings
    
ERA5_VARS = [
        "air_pressure_at_mean_sea_level",
        "air_temperature_at_2_metres",
        "air_temperature_at_2_metres_1hour_Maximum",
        "air_temperature_at_2_metres_1hour_Minimum",
        "dew_point_temperature_at_2_metres",
        "eastward_wind_at_100_metres",
        "eastward_wind_at_10_metres",
        "integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation",
        "lwe_thickness_of_surface_snow_amount",
        "northward_wind_at_100_metres",
        "northward_wind_at_10_metres",
        "precipitation_amount_1hour_Accumulation",
        "sea_surface_temperature",
        "sea_surface_wave_from_direction",
        "sea_surface_wave_mean_period",
        "sea_surface_wind_wave_from_direction",
        "significant_height_of_wind_and_swell_waves",
        "snow_density",
        "surface_air_pressure",        
    ]

def get_era5_daily(var,date_from_arg,date_to_arg=None,reduce_func=None,cache_dir='era5'):
    ''' 
    Return daily values from one of the ERA5 data vars 
    NOTE: The optional reduce_func arg lets you specify a function to apply to each day's worth of data.  
    The default is np.mean, which computes daily average.  To get a sum, use np.sum
    '''
    # Massage input data
    assert var in ERA5_VARS, "var must be one of [{}] (got {})".format(','.join(ERA5_VARS), var)
    if not os.path.exists(cache_dir): os.mkdir(cache_dir)
    if reduce_func is None:
        reduce_func = np.mean
    if type(date_from_arg)==str: date_from_arg = parse(date_from_arg)
    if type(date_to_arg)  ==str: date_to_arg   = parse(date_to_arg)
    if date_to_arg is None: date_to_arg = date_from_arg
    # Make sure our dates are in the correct order
    from_date = min(date_from_arg,date_to_arg)
    to_date   = max(date_from_arg,date_to_arg)
    # Download ERA5 files to local cache if they don't already exist
    client = None # Boto client (if needed)
    local_files=[] # Will hold list of local filenames
    Y,M = from_date.year, from_date.month # Loop vars
    loop_end = to_date.year*12+to_date.month # Loop sentinel
    while Y*12+M<=loop_end:
        local_file = os.path.join(cache_dir,"{Y:04}_{M:02}_{var}.nc".format(Y=Y,M=M,var=var))
        data_key = "{Y:04}/{M:02}/data/{var}.nc".format(Y=Y,M=M,var=var)
        if not os.path.isfile(local_file): # check if file already exists (TODO: move to temp?) (TODO: catch failed download)
            if client is None:
                client = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))
            client.download_file('era5-pds', data_key, local_file)
        local_files.append(local_file)
        if M==12: Y+=1; M=1
        else: M+=1
    # Load and merge the locally-cached ERA5 data from the list of filenames
    date_slice = slice(str(from_date.date()), str(to_date.date())) # I do this to INCLUDE the whole end date, not just 00:00

    def prepro(ds):
        if 'time0' in ds.dims: ds=ds.rename({"time0":"time"})
        if 'time1' in ds.dims: ds=ds.rename({"time1":"time"}) # This should INTENTIONALLY error if both times are defined
        return ds[[var]].sel(time=date_slice).resample(time='1D').reduce(reduce_func)
    return xr.open_mfdataset(local_files,combine='by_coords',compat='equals',preprocess=prepro,parallel=True)

def era5_area_crop(ds,lat,lon):
    # Handle single value lat/lon args by wrapping them in lists
    try: min(lat)
    except TypeError: lat=[lat]
    try: min(lon)
    except TypeError: lon=[lon]
    # We do NOT handle longitude wraparound - user should merge multiple calls to this function instead.
    # Technically the code will work with values outside the range, it will just round to the closest within range.
    # Since rounding -180 up to 0 may not be the expected behavior, issue warnings if args outside range.
    if min(lat) < ds.lat.min() or max(lat) > ds.lat.max():
        warnings.warn("Lats must be in range {} .. {}.  Got: {}".format(ds.lat.min().values,ds.lat.max().values,lat))
    if min(lon) < ds.lon.min() or max(lon) > ds.lon.max():
        warnings.warn("Lons must be in range {} .. {}.  Got: {}".format(ds.lon.min().values,ds.lon.max().values,lon))
    # Find existing coords between min&max
    lats = ds.lat[np.logical_and(ds.lat>=min(lat),ds.lat<=max(lat))].values 
    # If there was nothing between, just plan to grab closest
    if len(lats)==0: 
        lats = np.unique(ds.lat.sel(lat=np.array(lat), method="nearest"))        
    lons = ds.lon[np.logical_and(ds.lon>=min(lon),ds.lon<=max(lon))].values
    if len(lons)==0: 
        lons = np.unique(ds.lon.sel(lon=np.array(lon), method="nearest"))        
    return ds.sel(lat=lats,lon=lons)


def era5_area_nearest(ds, lat, lon):
    # alternative to the above crop method
    # snap to nearest data grid
    test = ds.sel(lat=lat, lon=lon, method='nearest')
    lat_range = slice(test.lat.values[0], test.lat.values[1])
    lon_range = slice(test.lon.values[0], test.lon.values[1])
    return ds.sel(lat=lat_range, lon=lon_range)

    
def load_era5(var, lat, lon, time, grid='nearest', **kwargs):
    ds = get_era5_daily(var,time[0],time[1], **kwargs)
    if grid == 'nearest': return era5_area_nearest(ds, lat, lon).compute()
    else: return era5_area_crop(ds, lat, lon).compute()

