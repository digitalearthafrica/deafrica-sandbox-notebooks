## deafrica_datahandling.py
'''
Description: This file contains a set of python functions for handling 
Digital Earth Africa data.

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
Github: https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues/new

Functions included:
    load_masked_usgs
    load_masked_FC
    array_to_geotiff
    mostcommon_utm
    download_unzip
    wofs_fuser
    dilate

Last modified: October 2019

'''

# Import required packages
import os
import gdal
import requests
import zipfile
import warnings
import numpy as np
import xarray as xr
from collections import Counter
from datacube.storage import masking
from scipy.ndimage import binary_dilation

    
def load_ard(dc,
             products=None,
             min_gooddata=0.0,
             quality_flags=None,
             mask_pixel_quality=True,
             mask_invalid_data=True,
             ls7_slc_off=True,
             product_metadata=False,
             **dcload_kwargs):
    '''
    Loads USGS Landsat Collection 1 and Collection 2 data for multiple 
    satellites (i.e. Landsat 5, 7, 8), and returns a single masked 
    xarray dataset containing only observations that contain greater 
    than a given proportion of good quality pixels. This can be used 
    to extract clean time series of observations that are not affected 
    by cloud, for example as an input to the `animated_timeseries` 
    function from `deafrica-sandbox-notebooks/deafrica_plotting`.
    
    The proportion of good quality pixels is calculated by summing the 
    pixels flagged as good quality in the product's pixel quality band 
    (i.e. 'pixel_qa' for USGS Collection 1, and 'quality_l2_aerosol' for
    USGS Collection 2). By default non-cloudy or non-shadowed pixels 
    are considered as good data, but this can be customised using the 
    `quality_flags` parameter.
    
    MEMORY ISSUES: For large data extractions, it can be advisable to 
    set `mask_pixel_quality=False`. The masking step coerces all 
    numeric values to float32 when NaN values are inserted into the 
    array, potentially causing your data to use twice the memory. 
    Be aware that the resulting arrays will contain invalid values 
    which may affect future analyses.
    
    Last modified: Feb 2020
    
    Parameters
    ----------  
    dc : datacube Datacube object
        The Datacube to connect to, i.e. `dc = datacube.Datacube()`.
        This allows you to also use development datacubes if required.    
    products : list
        A list of product names to load data from. Valid options for 
        USGS Collection 1 are ['ls5_usgs_sr_scene', 'ls7_usgs_sr_scene', 
        'ls8_usgs_sr_scene'] and for USGS Collection 2 are 
        ['usgs_ls5t_level2_2', 'usgs_ls7e_level2_2', 'usgs_ls8c_level2_2'].
    min_gooddata : float, optional
        An optional float giving the minimum percentage of good quality 
        pixels required for a satellite observation to be loaded. 
        Defaults to 0.0 which will return all observations regardless of
        pixel quality (set to e.g. 0.99 to return only observations with
        more than 99% good quality pixels).
    quality_flags : dict, optional
        An optional dictionary that is used to generate a good quality 
        pixel mask from the selected product's pixel quality band (i.e. 
        'pixel_qa' for USGS Collection 1, and 'quality_l2_aerosol' for
        USGS Collection 2). This mask is used for both masking out low
        quality pixels (e.g. cloud or shadow), and for dropping 
        observations entirely based on the above `min_gooddata` 
        calculation. Default is None, which will apply the following mask 
        for USGS Collection 1: `{'cloud': 'no_cloud', 'cloud_shadow': 
        'no_cloud_shadow', 'nodata': False}`, and for USGS Collection 2:
        `{'cloud_shadow': 'not_cloud_shadow', 'cloud_or_cirrus': 
        'not_cloud_or_cirrus', 'nodata': False}.
    mask_pixel_quality : bool, optional
        An optional boolean indicating whether to apply the good data 
        mask to all observations that were not filtered out for having 
        less good quality pixels than `min_gooddata`. E.g. if 
        `min_gooddata=0.99`, the filtered observations may still contain 
        up to 1% poor quality pixels. The default of False simply 
        returns the resulting observations without masking out these 
        pixels; True masks them out and sets them to NaN using the good 
        data mask. This will convert numeric values to float32 which can 
        cause memory issues, set to False to prevent this.
    mask_invalid_data : bool, optional
        An optional boolean indicating whether invalid -999 (or 0 for C2)
        nodata values should be replaced with NaN. These invalid values can be
        caused by missing data along the edges of scenes, or terrain 
        effects. Be aware that masking out invalid values 
        will convert all numeric values to floating point values when 
        values are replaced with NaN, which can cause memory issues.
    ls7_slc_off : bool, optional
        An optional boolean indicating whether to include data from 
        after the Landsat 7 SLC failure (i.e. SLC-off). Defaults to 
        True, which keeps all Landsat 7 observations > May 31 2003. 
    product_metadata : bool, optional
        An optional boolean indicating whether to return the dataset 
        with a `product` variable that gives the name of the product 
        that each observation in the time series came from (e.g. 
        'usgs_ls8c_level2_2'). Defaults to False.
    **dcload_kwargs : 
        A set of keyword arguments to `dc.load` that define the 
        spatiotemporal query used to extract data. This can include `x`,
        `y`, `time`, `resolution`, `resampling`, `group_by`, `crs`
        etc, and can either be listed directly in the `load_ard` call 
        (e.g. `x=(150.0, 151.0)`), or by passing in a query kwarg 
        (e.g. `**query`). For a full list of possible options, see: 
        https://datacube-core.readthedocs.io/en/latest/dev/api/generate/datacube.Datacube.load.html          
        
    Returns
    -------
    combined_ds : xarray Dataset
        An xarray dataset containing only satellite observations that 
        contains greater than `min_gooddata` proportion of good quality 
        pixels.   
        
    '''
    
    # Due to possible bug in xarray 0.13.0, define temporary function 
    # which converts dtypes in a way that preserves attributes
    def astype_attrs(da, dtype=np.float32):
        '''
        Loop through all data variables in the dataset, record 
        attributes, convert to float32, then reassign attributes. If 
        the data variable cannot be converted to float32 (e.g. for a
        non-numeric dtype like strings), skip and return the variable 
        unchanged.
        '''
        
        try:            
            da_attr = da.attrs
            da = da.astype(dtype)
            da = da.assign_attrs(**da_attr)
            return da
        
        except ValueError:        
            return da
    
    # Determine if lazy loading is required
    lazy_load = 'dask_chunks' in dcload_kwargs
    
    # List of valid USGS Collection 1 products
    c1_products = ['ls5_usgs_sr_scene',
                   'ls7_usgs_sr_scene',
                   'ls8_usgs_sr_scene']

    # List of valid USGS Collection 2 products
    c2_products = ['usgs_ls5t_level2_2', 
                   'usgs_ls7e_level2_2', 
                   'usgs_ls8c_level2_2']
    
    # Verify that products were provided
    if not products:
        raise ValueError(f'Please provide a list of product names '
                         f'to load data from. Valid options include '
                         f'{c1_products} and {c2_products}')
        
    # Verify that all provided products are valid
    not_in_list = [i for i in products if 
                   i not in c1_products + c2_products]
    if not_in_list:
        raise ValueError(f'The product(s) {not_in_list} are not '
                         f'supported by this function. Valid options '
                         f'include {c1_products} and {c2_products}')    
    
    # Warn user if they combine lazy load with min_gooddata
    if (min_gooddata > 0.0) & lazy_load:
                warnings.warn("Setting 'min_gooddata' percentage to > 0.0 "
                              "will cause dask arrays \n to compute when "
                              "loading pixel-quality data to calculate "
                              "'good pixel' percentage. This will "
                              "significantly slow the return of your dataset.")
    
    # Create a list to hold data for each product
    product_data = []

    # Iterate through each requested product
    for product in products:

        try:

            print(f'Loading {product} data')
            
            # Set quality band according to collection
            if product in c2_products:
                print('    Using pixel quality parameters for USGS Collection 2')
                quality_band = 'quality_l2_aerosol'
            elif product in c1_products:
                print('    Using pixel quality parameters for USGS Collection 1')
                quality_band = 'pixel_qa'

            # Set quality flags according to collection
            if (product in c2_products) and not quality_flags:
                quality_flags_prod = {'cloud_shadow': 'not_cloud_shadow',
                                      'cloud_or_cirrus': 'not_cloud_or_cirrus',
                                      'nodata': False}
            elif (product in c1_products) and not quality_flags:
                quality_flags_prod = {'cloud': 'no_cloud',
                                      'cloud_shadow': 'no_cloud_shadow',
                                      'nodata': False}
            
            elif quality_flags:
                quality_flags_prod = quality_flags
                
            # If `measurements` are specified but do not include fmask, add it
            if (('measurements' in dcload_kwargs) and 
                (quality_band not in dcload_kwargs['measurements'])):
                dcload_kwargs['measurements'].append(quality_band)

            # Load data
            try:
                ds = dc.load(product=f'{product}',
                             **dcload_kwargs)
            
            except KeyError as e:
                raise ValueError(f'Band {e} does not exist in this product. '
                                 f'Verify all requested `measurements` exist '
                                 f'in {products}')
            
            # Keep a record of the original number of observations
            total_obs = len(ds.time)

            # Remove Landsat 7 SLC-off observations if ls7_slc_off=False
            if not ls7_slc_off and product in ['ls7_usgs_sr_scene', 
                                               'usgs_ls7e_level2_2']:
                print('    Ignoring SLC-off observations for ls7')
                ds = ds.sel(time=ds.time < np.datetime64('2003-05-30'))
                
            # Identify all pixels not affected by cloud/shadow/invalid
            good_quality = masking.make_mask(ds[quality_band], 
                                             **quality_flags_prod)
            
            # The good data percentage calculation has to load in all `fmask`
            # data, which can be slow. If the user has chosen no filtering 
            # by using the default `min_gooddata = 0`, we can skip this step 
            # completely to save processing time
            if min_gooddata > 0.0:

                # Compute good data for each observation as % of total pixels
                data_perc = (good_quality.sum(axis=1).sum(axis=1) / 
                    (good_quality.shape[1] * good_quality.shape[2]))

                # Filter by `min_gooddata` to drop low quality observations
                ds = ds.sel(time=data_perc >= min_gooddata)
                print(f'    Filtering to {len(ds.time)} '
                      f'out of {total_obs} observations')

            # Optionally apply pixel quality mask to observations remaining 
            # after the filtering step above to mask out all remaining
            # bad quality pixels
            if mask_pixel_quality & (len(ds.time) > 0):
                print('    Applying pixel quality mask')
                
                # First change dtype to float32, then mask out values using
                # `.where()`. By casting to float32, we prevent `.where()` 
                # from automatically casting to float64, using 2x the memory.
                # We need to do this by applying a custom function to every
                # variable in the dataset instead of using `.astype()`, due 
                # to a possible bug in xarray 0.13.0 that drops attributes 
                ds = ds.apply(astype_attrs, dtype=np.float32, keep_attrs=True)
                ds = ds.where(good_quality)

            # Optionally add satellite/product name as a new variable
            if product_metadata:
                ds['product'] = xr.DataArray(
                    [product] * len(ds.time), [('time', ds.time)])

            # If any data was returned, add result to list
            if len(ds.time) > 0:
                product_data.append(ds.drop(quality_band))

        # If  AttributeError due to there being no pixel quality variable 
        # in the dataset, skip this product and move on to the next
        except AttributeError:
            print(f'    No data for {product}')

    # If any data was returned above, combine into one xarray
    if (len(product_data) > 0):

        # Concatenate results and sort by time
        try:
            print(f'Combining and sorting data')
            combined_ds = xr.concat(product_data, dim='time').sortby('time')
            
        except KeyError as e:
            raise ValueError(f'The requested products {products} contain '
                             f'bands with non-matching names (e.g. {e}). Please '
                             f'select products with identical band names.')
        
        # If `lazy_load` is True, return data as a dask array without
        # actually loading it in
        if lazy_load:
            print(f'    Returning {len(combined_ds.time)} observations'
                  ' as a dask array')
            return combined_ds

        else:
            print(f'    Returning {len(combined_ds.time)} observations ')
            return combined_ds.compute()

    # If no data was returned:
    else:
        print('No data returned for query')
        return None


def load_masked_FC(dc,
                   products=None,
                   min_gooddata=0.0,
                   quality_flags=None,
                   mask_pixel_quality=True,
                   mask_invalid_data=True,
                   ls7_slc_off=True,
                   product_metadata=False,
                   **dcload_kwargs):
    '''
    Loads Fractional Cover, calculated from the 
    USGS Landsat Collection 2 data for multiple 
    satellites (i.e. Landsat 5, 7, 8), and returns a single masked 
    xarray dataset containing only observations that contain greater 
    than a given proportion of good quality pixels. This can be used 
    to extract clean time series of observations that are not affected 
    by cloud, for example as an input to the `animated_timeseries` 
    function from `deafrica-sandbox-notebooks/deafrica_plotting`.

    The proportion of good quality pixels is calculated by summing the 
    pixels flagged as good quality in the product's pixel quality band 
    (i.e. 'pixel_qa' for USGS Collection 1, and 'quality_l2_aerosol' for
    USGS Collection 2). By default non-cloudy or non-shadowed pixels 
    are considered as good data, but this can be customised using the 
    `quality_flags` parameter.

    MEMORY ISSUES: For large data extractions, it can be advisable to 
    set `mask_pixel_quality=False`. The masking step coerces all 
    numeric values to float32 when NaN values are inserted into the 
    array, potentially causing your data to use twice the memory. 
    Be aware that the resulting arrays will contain invalid values 
    which may affect future analyses.

    Last modified: October 2019

    Parameters
    ----------  
    dc : datacube Datacube object
        The Datacube to connect to, i.e. `dc = datacube.Datacube()`.
        This allows you to also use development datacubes if required.    
    products : list
        A list of product names to load data from. Valid options for 
         USGS Collection 2 are ['ga_ls5t_fractional_cover_2',
         'ga_ls7e_fractional_cover_2', 'ga_ls8c_fractional_cover_2'].
    min_gooddata : float, optional
        An optional float giving the minimum percentage of good quality 
        pixels required for a satellite observation to be loaded. 
        Defaults to 0.0 which will return all observations regardless of
        pixel quality (set to e.g. 0.99 to return only observations with
        more than 99% good quality pixels).
    quality_flags : dict, optional
        An optional dictionary that is used to generate a good quality 
        pixel mask from the selected product's pixel quality band (i.e.
        'quality_l2_aerosol' for USGS Collection 2). This mask is used for 
        both masking out low quality pixels (e.g. cloud or shadow), and for dropping 
        observations entirely based on the above `min_gooddata` 
        calculation. Default is None, which will apply the following mask 
        for for USGS Collection 2:
        `{'cloud_shadow': 'not_cloud_shadow', 'cloud_or_cirrus': 
        'not_cloud_or_cirrus', 'nodata': False}.
    mask_pixel_quality : bool, optional
        An optional boolean indicating whether to apply the good data 
        mask to all observations that were not filtered out for having 
        less good quality pixels than `min_gooddata`. E.g. if 
        `min_gooddata=0.99`, the filtered observations may still contain 
        up to 1% poor quality pixels. The default of False simply 
        returns the resulting observations without masking out these 
        pixels; True masks them out and sets them to NaN using the good 
        data mask. This will convert numeric values to float32 which can 
        cause memory issues, set to False to prevent this.
    ls7_slc_off : bool, optional
        An optional boolean indicating whether to include data from 
        after the Landsat 7 SLC failure (i.e. SLC-off). Defaults to 
        True, which keeps all Landsat 7 observations > May 31 2003. 
    product_metadata : bool, optional
        An optional boolean indicating whether to return the dataset 
        with a `product` variable that gives the name of the product 
        that each observation in the time series came from (e.g. 
        'ga_ls8c_fractional_cover_2'). Defaults to False.
    **dcload_kwargs : 
        A set of keyword arguments to `dc.load` that define the 
        spatiotemporal query used to extract data. This can include `x`,
        `y`, `time`, `resolution`, `resampling`, `group_by`, `crs`
        etc, and can either be listed directly in the `load_ard` call 
        (e.g. `x=(150.0, 151.0)`), or by passing in a query kwarg 
        (e.g. `**query`). For a full list of possible options, see: 
        https://datacube-core.readthedocs.io/en/latest/dev/api/generate/datacube.Datacube.load.html          

    Returns
    -------
    combined_ds : xarray Dataset
        An xarray dataset containing only satellite observations that 
        contains greater than `min_gooddata` proportion of good quality 
        pixels.   

    '''
    def removekey(d, key):
        '''
        funtion to remove 'measurements' from
        the dcload_kwargs dictionary so they dont
        conflict with loading the correct measurements
        for the cloud dataset
        '''
        r = dict(d)
        del r[key]
        return r
            
    # Due to possible bug in xarray 0.13.0, define temporary function
    # which converts dtypes in a way that preserves attributes
    def astype_attrs(da, dtype=np.float32):
        '''
        Loop through all data variables in the dataset, record 
        attributes, convert to float32, then reassign attributes. If 
        the data variable cannot be converted to float32 (e.g. for a
        non-numeric dtype like strings), skip and return the variable 
        unchanged.
        '''

        try:
            da_attr = da.attrs
            da = da.astype(dtype)
            da = da.assign_attrs(**da_attr)
            return da

        except ValueError:
            return da
    
    # Determine if lazy loading is required
    lazy_load = 'dask_chunks' in dcload_kwargs
    
    # List of valid USGS Collection 2 products
    c2_products = ['ga_ls5t_fractional_cover_2',
                   'ga_ls7e_fractional_cover_2',
                   'ga_ls8c_fractional_cover_2']

    # Verify that products were provided
    if not products:
        raise ValueError(f'Please provide a list of product names '
                         f'to load data from. Valid options include '
                         f'{c2_products}')

    # Verify that all provided products are valid
    not_in_list = [i for i in products if
                   i not in c2_products]
    if not_in_list:
        raise ValueError(f'The product(s) {not_in_list} are not '
                         f'supported by this function. Valid options '
                         f'include {c2_products}')

    # Warn user if they combine lazy load with min_gooddata
    if (min_gooddata > 0.0) & lazy_load:
        warnings.warn("Setting 'min_gooddata' percentage to > 0.0 "
                      "will cause dask arrays \n to compute when "
                      "loading pixel-quality data to calculate "
                      "'good pixel' percentage. This will "
                      "significantly slow the return of your dataset.")

    # Create a list to hold data for each product
    product_data = []

    # Iterate through each requested product
    for product in products:

        try:

            print(f'Loading {product} data')

            # Set quality band according to collection
            if product in c2_products:
                print('    Using pixel quality parameters for USGS Collection 2')
                quality_band = 'quality_l2_aerosol'

            # Set quality flags according to collection
            if (product in c2_products) and not quality_flags:
                quality_flags_prod = {'cloud_shadow': 'not_cloud_shadow',
                                      'cloud_or_cirrus': 'not_cloud_or_cirrus',
                                      'nodata': False}
            elif quality_flags:
                quality_flags_prod = quality_flags

            # Load data
            ds = dc.load(product=f'{product}',
                         **dcload_kwargs)

            # Keep a record of the original number of observations
            total_obs = len(ds.time)

            # Remove Landsat 7 SLC-off observations if ls7_slc_off=False
            if not ls7_slc_off and product in ['ga_ls7e_fractional_cover_2']:
                print(' Ignoring SLC-off observations for ls7')
                ds = ds.sel(time=ds.time < np.datetime64('2003-05-30'))
            
            #remove 'measurements' so it doesn't conflict with loading
            #clouds datasets
            if 'measurements' in dcload_kwargs:
                cloud_kwargs = removekey(dcload_kwargs, 'measurements')
            
            # loud the clouds dataset
            if product == 'ga_ls8c_fractional_cover_2':
                clouds = dc.load(product='usgs_ls8c_level2_2',
                                 measurements=['quality_l2_aerosol'],
                                 **cloud_kwargs)

            elif product == 'ga_ls7e_fractional_cover_2':
                clouds = dc.load(product='usgs_ls7e_level2_2',
                                 measurements=['quality_l2_aerosol'],
                                 **cloud_kwargs)

            elif product == 'ga_ls5t_fractional_cover_2':
                clouds = dc.load(product='usgs_ls5t_level2_2',
                                 measurements=['quality_l2_aerosol'],
                                 **cloud_kwargs)

            # Identify all pixels not affected by cloud/shadow/invalid
            good_quality = masking.make_mask(clouds[quality_band],
                                             **quality_flags_prod)

            # The good data percentage calculation has to load in all `fmask`
            # data, which can be slow. If the user has chosen no filtering
            # by using the default `min_gooddata = 0`, we can skip this step
            # completely to save processing time
            if min_gooddata > 0.0:

                # Compute good data for each observation as % of total pixels
                data_perc = (good_quality.sum(axis=1).sum(axis=1) /
                             (good_quality.shape[1] * good_quality.shape[2]))

                # Filter by `min_gooddata` to drop low quality observations
                ds = ds.sel(time=data_perc >= min_gooddata)
                print(f'    Filtering to {len(ds.time)} '
                      f'out of {total_obs} observations')

            # Optionally apply pixel quality mask to observations remaining
            # after the filtering step above to mask out all remaining
            # bad quality pixels
            if mask_pixel_quality & (len(ds.time) > 0):
                print('    Applying pixel quality mask')

                # First change dtype to float32, then mask out values using
                # `.where()`. By casting to float32, we prevent `.where()`
                # from automatically casting to float64, using 2x the memory.
                # We need to do this by applying a custom function to every
                # variable in the dataset instead of using `.astype()`, due
                # to a possible bug in xarray 0.13.0 that drops attributes
                ds = ds.apply(astype_attrs, dtype=np.float32, keep_attrs=True)
                ds = ds.where(good_quality)

            # Optionally add satellite/product name as a new variable
            if product_metadata:
                ds['product'] = xr.DataArray(
                    [product] * len(ds.time), [('time', ds.time)])

            # If any data was returned, add result to list
            if len(ds.time) > 0:
                product_data.append(ds)

        # If  AttributeError due to there being no pixel quality variable
        # in the dataset, skip this product and move on to the next
        except AttributeError:
            print(f'    No data for {product}')

    # If any data was returned above, combine into one xarray
    if (len(product_data) > 0):

        # Concatenate results and sort by time
        try:
            print(f'Combining and sorting data')
            combined_ds = xr.concat(product_data, dim='time').sortby('time')

        except KeyError as e:
            raise ValueError(f'The requested products {products} contain '
                             f'bands with non-matching names (e.g. {e}). Please '
                             f'select products with identical band names.')

        # If `lazy_load` is True, return data as a dask array without
        # actually loading it in
        if lazy_load:
            print(f'    Returning {len(combined_ds.time)} observations'
                  ' as a dask array')
            return combined_ds

        else:
            print(f'    Returning {len(combined_ds.time)} observations ')
            return combined_ds.compute()

    # If no data was returned:
    else:
        print('No data returned for query')
        return None
    
    
def array_to_geotiff(fname, data, geo_transform, projection,
                     nodata_val=0, dtype=gdal.GDT_Float32):
    """
    Create a single band GeoTIFF file with data from an array. 
    
    Because this works with simple arrays rather than xarray datasets 
    from DEA, it requires geotransform info ("(upleft_x, x_size, 
    x_rotation, upleft_y, y_rotation, y_size)") and projection data 
    (in "WKT" format) for the output raster. These are typically 
    obtained from an existing raster using the following GDAL calls:
    
        import gdal
        gdal_dataset = gdal.Open(raster_path)
        geotrans = gdal_dataset.GetGeoTransform()
        prj = gdal_dataset.GetProjection()
    
    ...or alternatively, directly from an xarray dataset:
    
        geotrans = xarraydataset.geobox.transform.to_gdal()
        prj = xarraydataset.geobox.crs.wkt
    
    Parameters
    ----------     
    fname : str
        Output geotiff file path including extension
    data : numpy array
        Input array to export as a geotiff    
    geo_transform : tuple 
        Geotransform for output raster; e.g. "(upleft_x, x_size, 
        x_rotation, upleft_y, y_rotation, y_size)"
    projection : str
        Projection for output raster (in "WKT" format)
    nodata_val : int, optional
        Value to convert to nodata in the output raster; default 0
    dtype : gdal dtype object, optional
        Optionally set the dtype of the output raster; can be 
        useful when exporting an array of float or integer values. 
        Defaults to gdal.GDT_Float32
        
    """

    # Set up driver
    driver = gdal.GetDriverByName('GTiff')

    # Create raster of given size and projection
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, dtype)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)

    # Write data to array and set nodata values
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(nodata_val)

    # Close file
    dataset = None


def mostcommon_crs(dc, product, query):    
    """
    Takes a given query and returns the most common CRS for observations
    returned for that spatial extent. This can be useful when your study
    area lies on the boundary of two UTM zones, forcing you to decide
    which CRS to use for your `output_crs` in `dc.load`.
    
    Parameters
    ----------     
    dc : datacube Datacube object
        The Datacube to connect to, i.e. `dc = datacube.Datacube()`.
        This allows you to also use development datacubes if required.   
    product : str
        A product name to load CRSs from
    query : dict
        A datacube query including x, y and time range to assess for the
        most common CRS
        
    Returns
    -------
    A EPSG string giving the most common CRS from all datasets returned
    by the query above
    
    """
    
    # List of matching products
    matching_datasets = dc.find_datasets(product=product, **query)
    
    # Extract all CRSs
    crs_list = [str(i.crs) for i in matching_datasets]    
   
    # Identify most common CRS
    crs_counts = Counter(crs_list)    
    crs_mostcommon = crs_counts.most_common(1)[0][0]

    # Warn user if multiple CRSs are encountered
    if len(crs_counts.keys()) > 1:

        warnings.warn(f'Multiple UTM zones {list(crs_counts.keys())} '
                      f'were returned for this query. Defaulting to '
                      f'the most common zone: {crs_mostcommon}', 
                      UserWarning)
    
    return crs_mostcommon


def download_unzip(url,
                   output_dir=None,
                   remove_zip=True):
    """
    Downloads and unzips a .zip file from an external URL to a local
    directory.
    
    Parameters
    ----------     
    url : str
        A string giving a URL path to the zip file you wish to download
        and unzip
    output_dir : str, optional
        An optional string giving the directory to unzip files into. 
        Defaults to None, which will unzip files in the current working 
        directory
    remove_zip : bool, optional
        An optional boolean indicating whether to remove the downloaded
        .zip file after files are unzipped. Defaults to True, which will
        delete the .zip file.  
    
    """
    
    # Get basename for zip file
    zip_name = os.path.basename(url)
    
    # Raise exception if the file is not of type .zip
    if not zip_name.endswith('.zip'):
        raise ValueError(f'The URL provided does not point to a .zip '
                         f'file (e.g. {zip_name}). Please specify a '
                         f'URL path to a valid .zip file')
                         
    # Download zip file
    print(f'Downloading {zip_name}')
    r = requests.get(url)
    with open(zip_name, 'wb') as f:
        f.write(r.content)
        
    # Extract into output_dir
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:        
        zip_ref.extractall(output_dir)        
        print(f'Unzipping output files to: '
              f'{output_dir if output_dir else os.getcwd()}')
    
    # Optionally cleanup
    if remove_zip:        
        os.remove(zip_name)

        
def wofs_fuser(dest, src):
    """
    Fuse two WOfS water measurements represented as `ndarray`s.
    
    Note: this is a copy of the function located here:
    https://github.com/GeoscienceAustralia/digitalearthau/blob/develop/digitalearthau/utils.py
    """
    empty = (dest & 1).astype(np.bool)
    both = ~empty & ~((src & 1).astype(np.bool))
    dest[empty] = src[empty]
    dest[both] |= src[both]
    

def dilate(array, dilation=10, invert=True):
    """
    Dilate a binary array by a specified nummber of pixels using a 
    disk-like radial dilation.
    
    By default, invalid (e.g. False or 0) values are dilated. This is
    suitable for applications such as cloud masking (e.g. creating a 
    buffer around cloudy or shadowed pixels). This functionality can 
    be reversed by specifying `invert=False`.
    
    Parameters
    ----------     
    array : array
        The binary array to dilate.
    dilation : int, optional
        An optional integer specifying the number of pixels to dilate 
        by. Defaults to 10, which will dilate `array` by 10 pixels.
    invert : bool, optional
        An optional boolean specifying whether to invert the binary 
        array prior to dilation. The default is True, which dilates the
        invalid values in the array (e.g. False or 0 values).
        
    Returns
    -------
    An array of the same shape as `array`, with valid data pixels 
    dilated by the number of pixels specified by `dilation`.    
    """
    
    y, x = np.ogrid[
        -dilation : (dilation + 1),
        -dilation : (dilation + 1),
    ]
    
    # disk-like radial dilation
    kernel = (x * x) + (y * y) <= (dilation + 0.5) ** 2
    
    # If invert=True, invert True values to False etc
    if invert:        
        array = ~array
    
    return ~binary_dilation(array.astype(np.bool), 
                            structure=kernel.reshape((1,) + kernel.shape))
