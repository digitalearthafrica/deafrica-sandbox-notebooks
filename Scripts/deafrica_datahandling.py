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
    load_ard
    load_masked_FC
    array_to_geotiff
    mostcommon_utm
    download_unzip
    wofs_fuser
    dilate
    first
    last
    nearest

Last modified: March 2020

'''

# Import required packages
import os
from osgeo import gdal
import requests
import zipfile
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import pytz

from collections import Counter
from datacube.utils import masking
from scipy.ndimage import binary_dilation
from copy import deepcopy
import odc.algo
from random import randint
import numexpr as ne
import dask
import dask.array as da


def _dc_query_only(**kw):
    """ Remove load-only parameters, the rest can be passed to Query

    Returns
    =======

    dict of query parameters
    """

    def _impl(measurements=None,
              output_crs=None,
              resolution=None,
              resampling=None,
              skip_broken_datasets=None,
              dask_chunks=None,
              fuse_func=None,
              align=None,
              datasets=None,
              progress_cbk=None,
              group_by=None,
              **query):
        return query

    return _impl(**kw)


def _common_bands(dc, products):
    """
    Takes a list of products and returns a list of measurements/bands
    that are present in all products
    Returns
    -------
    List of band names
    """
    common = None
    bands = None

    for p in products:
        p = dc.index.products.get_by_name(p)
        if common is None:
            common = set(p.measurements)
            bands = list(p.measurements)
        else:
            common = common.intersection(set(p.measurements))
    return [band for band in bands if band in common]

    
def load_ard(dc,
             products=None,
             min_gooddata=0.0,
             pq_categories_s2=['vegetation','snow or ice',
                               'water','bare soils',
                               'unclassified', 'dark area pixels'],
             pq_categories_ls=None,
             mask_pixel_quality=True,
             ls7_slc_off=True,
             predicate=None,
             dtype='auto',
             scaling='raw',
             **kwargs):

    '''
    Loads and combines Landsat Collections 1 or 2, and Sentinel-2 for 
    multiple sensors (i.e. ls5t, ls7e and ls8c for Landsat; s2a and s2b for Sentinel-2), 
    optionally applies pixel quality masks, and drops time steps that 
    contain greater than a minimum proportion of good quality (e.g. non-
    cloudy or shadowed) pixels. 
    The function supports loading the following DE Africa products:
    
        ls5_usgs_sr_scene
        ls7_usgs_sr_scene
        ls8_usgs_sr_scene
        usgs_ls8c_level2_2
        ga_ls8c_fractional_cover_2
        s2_l2a

    Last modified: March 2020
    
    Parameters
    ----------
    dc : datacube Datacube object
        The Datacube to connect to, i.e. `dc = datacube.Datacube()`.
        This allows you to also use development datacubes if required.
    products : list
        A list of product names to load data from. Valid options are
        Landsat C1: ['ls5_usgs_sr_scene', 'ls7_usgs_sr_scene', 'ls8_usgs_sr_scene'],
        Landsat C2: ['usgs_ls8c_level2_2']
        Sentinel-2: ['s2_l2a']
    min_gooddata : float, optional
        An optional float giving the minimum percentage of good quality
        pixels required for a satellite observation to be loaded.
        Defaults to 0.0 which will return all observations regardless of
        pixel quality (set to e.g. 0.99 to return only observations with
        more than 99% good quality pixels).
    pq_categories_s2 : list, optional
        An optional list of Sentinel-2 Scene Classification Layer (SCL) names 
        to treat as good quality observations in the above `min_gooddata` 
        calculation. The default is ['vegetation','snow or ice','water',
        'bare soils','unclassified', 'dark area pixels'] which will return
        non-cloudy or non-shadowed land, snow, water, veg, and non-veg pixels.
    pq_categories_ls : dict, optional
        An optional dictionary that is used to generate a good quality 
        pixel mask from the selected USGS product's pixel quality band (i.e. 
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
        pixels; True masks them and sets them to NaN using the good data
        mask. This will convert numeric values to floating point values
        which can cause memory issues, set to False to prevent this.
    ls7_slc_off : bool, optional
        An optional boolean indicating whether to include data from
        after the Landsat 7 SLC failure (i.e. SLC-off). Defaults to
        True, which keeps all Landsat 7 observations > May 31 2003.
    predicate : function, optional
        An optional function that can be passed in to restrict the
        datasets that are loaded by the function. A filter function
        should take a `datacube.model.Dataset` object as an input (i.e.
        as returned from `dc.find_datasets`), and return a boolean.
        For example, a filter function could be used to return True on
        only datasets acquired in January:
        `dataset.time.begin.month == 1`
    dtype : string, optional
        An optional parameter that controls the data type/dtype that
        layers are coerced to after loading. Valid values: 'native', 
        'auto', 'float{16|32|64}'. When 'auto' is used, the data will be 
        converted to `float32` if masking is used, otherwise data will 
        be returned in the native data type of the data. Be aware that
        if data is loaded in its native dtype, nodata and masked 
        pixels will be returned with the data's native nodata value 
        (typically -999), not NaN. 
    scaling : str, optional
        If 'normalised', then surface reflectance values are scaled from
        their original values to 0-1.  If 'raw' then dataset is returned
        in its native scaling. WARNING: USGS Landsat Collection 2
        surface reflectance values have an offset so normliaed band indices 
        will return non-sensical results if setting scaling='raw'. 
    **kwargs :
        A set of keyword arguments to `dc.load` that define the
        spatiotemporal query used to extract data. This typically
        includes `measurements`, `x`, `y`, `time`, `resolution`,
        `resampling`, `group_by` and `crs`. Keyword arguments can
        either be listed directly in the `load_ard` call like any
        other parameter (e.g. `measurements=['nbart_red']`), or by
        passing in a query kwarg dictionary (e.g. `**query`). For a
        list of possible options, see the `dc.load` documentation:
        https://datacube-core.readthedocs.io/en/latest/dev/api/generate/datacube.Datacube.load.html
        
    Returns
    -------
    combined_ds : xarray Dataset
        An xarray dataset containing only satellite observations that
        contains greater than `min_gooddata` proportion of good quality
        pixels.
        
    '''

    #########
    # Setup #
    #########
    # prevent function altering original query object
    kwargs = deepcopy(kwargs)

    # We deal with `dask_chunks` separately
    dask_chunks = kwargs.pop('dask_chunks', None)
    requested_measurements = kwargs.pop('measurements', None)

    # Warn user if they combine lazy load with min_gooddata
    if (min_gooddata > 0.0) and dask_chunks is not None:
        warnings.warn("Setting 'min_gooddata' percentage to > 0.0 "
                      "will cause dask arrays to compute when "
                      "loading pixel-quality data to calculate "
                      "'good pixel' percentage. This can "
                      "slow the return of your dataset.")
    
    # Verify that products were provided and determine if Sentinel-2
    # or Landsat data is being loaded
    if not products:
        raise ValueError(f'Please provide a list of product names '
                         f'to load data from.')
        
    elif all(['level2' in product for product in products]):
        product_type = 'c2'
    elif all(['sr' in product for product in products]):
        product_type = 'c1'
    elif all(['s2' in product for product in products]):
        product_type = 's2'
    elif all(['fractional_cover' in product for product in products]):
        product_type = 'fc'
                         
    # If `measurements` are specified but do not include pixel quality bands,
    #  add these to `measurements` according to collection
    if (product_type == 'c2') or (product_type == 'fc'):
        print('Using pixel quality parameters for USGS Collection 2')
        fmask_band = 'quality_l2_aerosol'
                        
    elif product_type == 'c1':
        print('Using pixel quality parameters for USGS Collection 1')
        fmask_band = 'pixel_qa'
    
    elif product_type == 's2':
        print('Using pixel quality parameters for Sentinel 2')
        fmask_band = 'SCL'
    
    measurements = requested_measurements.copy() if requested_measurements else None
    
    # Deal with "load all" case: pick a set of bands common across 
    # all products    
    if measurements is None:
        if product_type == 'fc':
            measurements = ['pv', 'npv', 'bs', 'ue']
        else:
            measurements = _common_bands(dc, products)
    
    # If `measurements` are specified but do not include pq, add.
    if measurements:
        #pass if FC
        if product_type == 'fc':
            pass
        else:
            if fmask_band not in measurements:
                measurements.append(fmask_band)
   
    # Get list of data and mask bands so that we can later exclude
    # mask bands from being masked themselves
    if product_type == 'fc':
        pass
    else:
        data_bands = [band for band in measurements if band not in (fmask_band)]
        mask_bands = [band for band in measurements if band not in data_bands]
    
    #################
    # Find datasets #
    #################l
    
    # Pull out query params only to pass to dc.find_datasets
    query = _dc_query_only(**kwargs)
    
    # Extract datasets for each product using subset of dcload_kwargs
    dataset_list = []
     
    # Get list of datasets for each product
    print('Finding datasets')
    for product in products:

        # Obtain list of datasets for product
        print(f'    {product}')
        datasets = dc.find_datasets(product=product, **query)
        
        # Remove Landsat 7 SLC-off observations if ls7_slc_off=False
        if not ls7_slc_off and product in ['ls7_usgs_sr_scene', 
                                           'usgs_ls7e_level2_2']:
            print('    Ignoring SLC-off observations for ls7')
            datasets = [i for i in datasets if i.time.begin <
                        datetime.datetime(2003, 5, 31, tzinfo=pytz.UTC)]

        # Add any returned datasets to list
        dataset_list.extend(datasets)       
    
    # Raise exception if no datasets are returned
    if len(dataset_list) == 0:
        raise ValueError("No data available for query: ensure that "
                         "the products specified have data for the "
                         "time and location requested")

    # If pedicate is specified, use this function to filter the list
    # of datasets prior to load
    if predicate:
        print(f'Filtering datasets using filter function')
        dataset_list = [ds for ds in dataset_list if predicate(ds)]
            
    # Raise exception if filtering removes all datasets
    if len(dataset_list) == 0:
        raise ValueError("No data available after filtering with "
                         "filter function")
    
    # load fmask from C2 for masking FC, and filter if required
    # NOTE: This works because only one sensor (ls8) has FC, if/when
    # FC is calculated for LS7, LS5, will need to move this section
    # into the for loop above.
    if product_type == 'fc':
              
        print('    PQ data from USGS C2')
        dataset_list_fc_pq = dc.find_datasets(product='usgs_ls8c_level2_2', **query)
        
        if predicate:
            print(f'Filtering datasets using filter function')
            dataset_list_fc_pq = [ds for ds in dataset_list_fc_pq if predicate(ds)]

    #############
    # Load data #
    #############
    # Note we always load using dask here so that
    # we can lazy load data before filtering by good data
    ds = dc.load(datasets=dataset_list,
                 measurements=measurements,
                 dask_chunks={} if dask_chunks is None else dask_chunks,
                 **kwargs)
   
    if product_type == 'fc':
        ds_fc_pq = dc.load(datasets=dataset_list_fc_pq,
                           dask_chunks={} if dask_chunks is None else dask_chunks,
                           **kwargs)
        
    ####################
    # Filter good data #
    ####################
    
    # need to distinguish between products due to different
    # pq band properties                     
    
    # collection 2 USGS or FC
    if (product_type == 'c2') or (product_type == 'fc'):
        if pq_categories_ls is None:
            quality_flags_prod = {'cloud_shadow': 'not_cloud_shadow',
                                  'cloud_or_cirrus': 'not_cloud_or_cirrus',
                                   'nodata': False}
        else:
            quality_flags_prod = pq_categories_ls
        
        if product_type == 'fc':
            pq_mask = masking.make_mask(ds_fc_pq[fmask_band], 
                                        **quality_flags_prod)
        else:
            pq_mask = masking.make_mask(ds[fmask_band], 
                                        **quality_flags_prod)
            
    # collection 1 USGS                    
    if product_type == 'c1':
        if pq_categories_ls is None:
            quality_flags_prod = {'cloud': 'no_cloud',
                                  'cloud_shadow': 'no_cloud_shadow',
                                   'nodata': False}
        else:
            quality_flags_prod = pq_categories_ls
            
        pq_mask = masking.make_mask(ds[fmask_band], 
                                    **quality_flags_prod)
    # sentinel 2                     
    if product_type == 's2':
        #currently broken for mask band values >=8
        #pq_mask = odc.algo.fmask_to_bool(ds[fmask_band],
        #                             categories=pq_categories_s2)
        flags_s2 = dc.list_measurements().loc[products[0]].loc[fmask_band]['flags_definition']['qa']['values']
        pq_mask = ds[fmask_band].isin([int(k) for k,v in flags_s2.items() if v in pq_categories_s2])
    
    # The good data percentage calculation has to load in all `fmask`
    # data, which can be slow. If the user has chosen no filtering
    # by using the default `min_gooddata = 0`, we can skip this step
    # completely to save processing time
    if min_gooddata > 0.0:

        # Compute good data for each observation as % of total pixels
        print('Counting good quality pixels for each time step')
        data_perc = (pq_mask.sum(axis=[1, 2], dtype='int32') /
                     (pq_mask.shape[1] * pq_mask.shape[2]))
        
        keep = data_perc >= min_gooddata
        
        # Filter by `min_gooddata` to drop low quality observations
        total_obs = len(ds.time)
        ds = ds.sel(time=keep)
        pq_mask = pq_mask.sel(time=keep)
        print(f'Filtering to {len(ds.time)} out of {total_obs} '
              f'time steps with at least {min_gooddata:.1%} '
              f'good quality pixels')
    
    ###############
    # Apply masks #
    ###############

    # Generate good quality data mask
    mask = None
    if mask_pixel_quality:
        print('Applying pixel quality/cloud mask')
        mask = pq_mask
    
    # Split into data/masks bands, as conversion to float and masking 
    # should only be applied to data bands    
    if product_type == 'fc':
        ds_data=ds
    else:
        ds_data = ds[data_bands]
        ds_masks = ds[mask_bands]

    # Mask data if either of the above masks were generated
    if mask is not None:  
            ds_data = odc.algo.keep_good_only(ds_data, where=mask)
    
    # Automatically set dtype to either native or float32 depending
    # on whether masking was requested
    if dtype == 'auto':
        dtype = 'native' if mask is None else 'float32'
    
    # Set nodata values using odc.algo tools to reduce peak memory
    # use when converting data dtype    
    if dtype != 'native':
        ds_data = odc.algo.to_float(ds_data, dtype=dtype)
    
    # Put data and mask bands back together
    if product_type == 'fc':
        attrs = ds.attrs
        ds = ds_data
        ds.attrs.update(attrs)
    else:
        attrs = ds.attrs
        ds = xr.merge([ds_data, ds_masks])
        ds.attrs.update(attrs)

    ###############
    # Return data #
    ###############
    
     # Drop bands not originally requested by user
    if requested_measurements:
        ds = ds[requested_measurements]
    
    # Scale data 0-1 if requested
    if scaling=='normalised':
        
        if product_type == 'c1':
            print("Re-scaling Landsat C1 data")
            not_sr_bands = ['pixel_qa','sr_aerosol','radsat_qa']
        
            for band in ds.data_vars:
                if band not in not_sr_bands:
                    ds[band]=ds[band]/10000

        if product_type == 's2':
            print("Re-scaling Sentinel-2 data")
            not_sr_bands = ['scl','qa','mask','water_vapour','aerosol_optical_thickness']
        
            for band in ds.data_vars:
                if band not in not_sr_bands:
                    ds[band]=ds[band]/10000   
    
    # Collection 2 Landsat raw values aren't useful so rescale,
    # need different factors for surface-temp and SR
    if product_type == 'c2':
        print("Re-scaling Landsat C2 data")
        not_sr_bands = ['thermal_radiance','upwell_radiance','upwell_radiance',
                        'atmospheric_transmittance','emissivity','emissivity_stdev',
                        'cloud_distance', 'quality_l2_aerosol','quality_l2_surface_temperature',
                        'quality_l1_pixel','quality_l1_radiometric_saturation','surface_temperature']
        
        for band in ds.data_vars:
        
            if band == 'surface_temperature':
                ds[band]=ds[band]*0.00341802 + 149.0 - 273.15
        
            if band not in not_sr_bands:
                ds[band]=ds[band]* 2.75e-5 - 0.2
            
    # If user supplied dask_chunks, return data as a dask array without
    # actually loading it in
    if dask_chunks is not None:
        print(f'Returning {len(ds.time)} time steps as a dask array')
        return ds
    else:
        print(f'Loading {len(ds.time)} time steps')
        return ds.compute()
    
    
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
    
    
    # prevent function altering dictionary kwargs
    query = deepcopy(query)
    
    # remove dask_chunks & align to prevent func failing
    if 'dask_chunks' in query:
        query.pop('dask_chunks', None)
 
    if 'align' in query:
        query.pop('align', None)
    
    # List of matching products    
    matching_datasets = dc.find_datasets(product=product, **query)
    
    # Extract all CRSs
    crs_list = [str(i.crs) for i in matching_datasets]    
   
    # If CRSs are returned
    if len(crs_list) > 0:

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
    
    else:
        
        raise ValueError(f'No CRS was returned as no data was found for '
                         f'the supplied product ({product}) and query. '
                         f'Please ensure that data is available for '
                         f'{product} for the spatial extents and time '
                         f'period specified in the query (e.g. by using '
                         f'the Data Cube Explorer for this datacube '
                         f'instance).')


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


def _select_along_axis(values, idx, axis):
    other_ind = np.ix_(*[np.arange(s) for s in idx.shape])
    sl = other_ind[:axis] + (idx,) + other_ind[axis:]
    return values[sl]


def first(array: xr.DataArray, dim: str, index_name: str = None) -> xr.DataArray:
    """
    Finds the first occuring non-null value along the given dimension.
    
    Parameters
    ----------
    array : xr.DataArray
         The array to search.
    dim : str
        The name of the dimension to reduce by finding the first non-null value.
    
    Returns
    -------
    reduced : xr.DataArray
        An array of the first non-null values.
        The `dim` dimension will be removed, and replaced with a coord of the 
        same name, containing the value of that dimension where the last value 
        was found.
    """
    axis = array.get_axis_num(dim)
    idx_first = np.argmax(~pd.isnull(array), axis=axis)
    reduced = array.reduce(_select_along_axis, idx=idx_first, axis=axis)
    reduced[dim] = array[dim].isel({dim: xr.DataArray(idx_first, dims=reduced.dims)})
    if index_name is not None:
        reduced[index_name] = xr.DataArray(idx_first, dims=reduced.dims)
    return reduced


def last(array: xr.DataArray, dim: str, index_name: str = None) -> xr.DataArray:
    """
    Finds the last occuring non-null value along the given dimension.
    
    Parameters
    ----------
    array : xr.DataArray
         The array to search.
    dim : str
        The name of the dimension to reduce by finding the last non-null value.
    index_name : str, optional
        If given, the name of a coordinate to be added containing the index
        of where on the dimension the nearest value was found.
    
    Returns
    -------
    reduced : xr.DataArray
        An array of the last non-null values.
        The `dim` dimension will be removed, and replaced with a coord of the 
        same name, containing the value of that dimension where the last value 
        was found.
    """
    axis = array.get_axis_num(dim)
    rev = (slice(None),) * axis + (slice(None, None, -1),)
    idx_last = -1 - np.argmax(~pd.isnull(array)[rev], axis=axis)
    reduced = array.reduce(_select_along_axis, idx=idx_last, axis=axis)
    reduced[dim] = array[dim].isel({dim: xr.DataArray(idx_last, dims=reduced.dims)})
    if index_name is not None:
        reduced[index_name] = xr.DataArray(idx_last, dims=reduced.dims)
    return reduced


def nearest(array: xr.DataArray, dim: str, target, index_name: str = None) -> xr.DataArray:
    """
    Finds the nearest values to a target label along the given dimension, for
    all other dimensions.
    
    E.g. For a DataArray with dimensions ('time', 'x', 'y')
    
        nearest_array = nearest(array, 'time', '2017-03-12')
        
    will return an array with the dimensions ('x', 'y'), with non-null values 
    found closest for each (x, y) pixel to that location along the time 
    dimension.
    
    The returned array will include the 'time' coordinate for each x,y pixel
    that the nearest value was found.
    
    Parameters
    ----------
    array : xr.DataArray
         The array to search.
    dim : str
        The name of the dimension to look for the target label.
    target : same type as array[dim]
        The value to look up along the given dimension.
    index_name : str, optional
        If given, the name of a coordinate to be added containing the index
        of where on the dimension the nearest value was found.
    
    Returns
    -------
    nearest_array : xr.DataArray
        An array of the nearest non-null values to the target label.
        The `dim` dimension will be removed, and replaced with a coord of the 
        same name, containing the value of that dimension closest to the
        given target label.
    """
    before_target = slice(None, target)
    after_target = slice(target, None)
    
    da_before = array.sel({dim: before_target})
    da_after = array.sel({dim: after_target})
        
    da_before = last(da_before, dim, index_name) if da_before[dim].shape[0] else None
    da_after = first(da_after, dim, index_name) if da_after[dim].shape[0] else None
    
    if da_before is None and da_after is not None:
        return da_after
    if da_after is None and da_before is not None:
        return da_before
    
    target = array[dim].dtype.type(target)
    is_before_closer = abs(target - da_before[dim]) < abs(target - da_after[dim])
    nearest_array = xr.where(is_before_closer, da_before, da_after)
    nearest_array[dim] = xr.where(is_before_closer, da_before[dim], da_after[dim])
    if index_name is not None:
        nearest_array[index_name] = xr.where(is_before_closer, 
                                             da_before[index_name], 
                                             da_after[index_name])
    return nearest_array

