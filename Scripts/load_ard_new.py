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
from copy import deepcopy
import odc.algo

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


def load_ard(dc,
             products=None,
             min_gooddata=0.0,
             pq_categories_s2=['vegetation','snow or ice','water',
                                'bare soils','unclassified'],
             pq_categories_ls=None,
             mask_pixel_quality=True,
             ls7_slc_off=True,
             filter_func=None,
             **extras):

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
    `fmask_categories` parameter.
    
    Last modified: February 2020
    Parameters
    ----------
    dc : datacube Datacube object
        The Datacube to connect to, i.e. `dc = datacube.Datacube()`.
        This allows you to also use development datacubes if required.
    products : list
        A list of product names to load data from. Valid options are
        ['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'] for Landsat,
        ['s2a_ard_granule', 's2b_ard_granule'] for Sentinel 2 Definitive,
        and ['s2a_nrt_granule', 's2b_nrt_granule'] for Sentinel 2 Near
        Real Time (on the DEA Sandbox only).
    min_gooddata : float, optional
        An optional float giving the minimum percentage of good quality
        pixels required for a satellite observation to be loaded.
        Defaults to 0.0 which will return all observations regardless of
        pixel quality (set to e.g. 0.99 to return only observations with
        more than 99% good quality pixels).
    pq_categories_s2 : list, optional
        An optional list of S2 Scene Classification Layer (SCL) names 
        to treat as good quality observations in the above `min_gooddata` 
        calculation. T The default is ['vegetation','snow or ice','water',
        'bare soils','unclassified'] which will return
        non-cloudy or shadowed land, snow, water, veg, and non-veg pixels.
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
    mask_contiguity : str or bool, optional
        An optional string or boolean indicating whether to mask out
        pixels missing data in any band (i.e. "non-contiguous" values).
        Although most missing data issues are resolved by
        `mask_invalid_data`, this step is important for generating
        clean and concistent composite datasets. The default
        is `mask_contiguity='nbart_contiguity'` which will set any
        pixels with non-contiguous values to NaN based on NBART data.
        If you are loading NBAR data instead, you should specify
        `mask_contiguity='nbar_contiguity'` instead. To ignore non-
        contiguous values completely, set `mask_contiguity=False`.
        Be aware that masking out non-contiguous values will convert
        all numeric values to floating point values when -999 values
        are replaced with NaN, which can cause memory issues.
    mask_dtype : numpy dtype, optional
        An optional parameter that controls the data type/dtype that
        layers are coerced to when when `mask_pixel_quality=True` or
        `mask_contiguity=True`. Defaults to `np.float32`, which uses
        approximately 1/2 the memory of `np.float64`.
    ls7_slc_off : bool, optional
        An optional boolean indicating whether to include data from
        after the Landsat 7 SLC failure (i.e. SLC-off). Defaults to
        True, which keeps all Landsat 7 observations > May 31 2003.
    filter_func : function, optional
        An optional function that can be passed in to restrict the
        datasets that are loaded by the function. A filter function
        should take a `datacube.model.Dataset` object as an input (i.e.
        as returned from `dc.find_datasets`), and return a boolean.
        For example, a filter function could be used to return True on
        only datasets acquired in January:
        `dataset.time.begin.month == 1`
    **extras :
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

    query = _dc_query_only(**extras)

    # We deal with `dask_chunks` separately
    dask_chunks = extras.pop('dask_chunks', None)
    requested_measurements = extras.pop('measurements', None)

    # Warn user if they combine lazy load with min_gooddata
    if (min_gooddata > 0.0) and dask_chunks is not None:
        warnings.warn("Setting 'min_gooddata' percentage to > 0.0 "
                      "will cause dask arrays to compute when "
                      "loading pixel-quality data to calculate "
                      "'good pixel' percentage. This can "
                      "slow the return of your dataset.")
    
    # Verify that products were provided
    if not products:
        raise ValueError(f'Please provide a list of product names '
                         f'to load data from. Valid options include '
                         f'{c1_products}, {c2_products} and {s2_products}')
        
    elif all(['level2' in product for product in products]):
        product_type = 'c2'
    elif all(['sr' in product for product in products]):
        product_type = 'c1'
    elif all(['s2' in product for product in products]):
        product_type = 's2'
                         
    # If `measurements` are specified but do not include pixel quality bands,
    #  add these to `measurements` according to collection
    if product_type == 'c2':
        print('Using pixel quality parameters for USGS Collection 2')
        fmask_band = 'quality_l2_aerosol'
                        
    elif product_type == 'c1':
        print('Using pixel quality parameters for USGS Collection 1')
        fmask_band = 'pixel_qa'
    
    elif product_type == 's2':
        print('Using pixel quality parameters for Sentinel 2')
        fmask_band = 'scl'
    
    print("pq band is: " + fmask_band)
    measurements = requested_measurements.copy() if requested_measurements else None
    
    print(measurements)                 
    
    if measurements:
        if fmask_band not in measurements:
            measurements.append(fmask_band)
    
    print(measurements)
    
    #################
    # Find datasets #
    #################

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
                        datetime.datetime(2003, 5, 31)]

        # Add any returned datasets to list
        dataset_list.extend(datasets)

    # Raise exception if no datasets are returned
    if len(dataset_list) == 0:
        raise ValueError("No data available for query: ensure that "
                         "the products specified have data for the "
                         "time and location requested")

    # If filter_func is specified, use this function to filter the list
    # of datasets prior to load
    if filter_func:
        print(f'Filtering datasets using filter function')
        dataset_list = [ds for ds in dataset_list if filter_func(ds)]

    # Raise exception if filtering removes all datasets
    if len(dataset_list) == 0:
        raise ValueError("No data available after filtering with "
                         "filter function")

    #############
    # Load data #
    #############

    # Note we always load using dask here so that
    # we can lazy load data before filtering by good data
    ds = dc.load(datasets=dataset_list,
                 measurements=measurements,
                 dask_chunks={} if dask_chunks is None else dask_chunks,
                 **extras)

    ###############
    # Apply masks #
    ###############
    #need to distinguish between products due to different
    # "fmask" band properties                     
    
    #collection 2 USGS
    if product_type == 'c2':
        if pq_categories_ls is None:
            quality_flags_prod = {'cloud_shadow': 'not_cloud_shadow',
                                  'cloud_or_cirrus': 'not_cloud_or_cirrus',
                                   'nodata': False}
        else:
            quality_flags_prod = pq_categories_ls

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
        pq_mask = odc.algo.fmask_to_bool(ds[fmask_band],
                                     categories=pq_categories_s2)

    # Generate good quality data mask
    mask = None
    if mask_pixel_quality:
        print('Applying pixel quality/cloud mask')
        mask = pq_mask

    # Mask data if either of the above masks were generated
    if mask is not None:
        ds = odc.algo.keep_good_only(ds, where=mask)

    ####################
    # Filter good data #
    ####################

    # The good data percentage calculation has to load in all `fmask`
    # data, which can be slow. If the user has chosen no filtering
    # by using the default `min_gooddata = 0`, we can skip this step
    # completely to save processing time
    if min_gooddata > 0.0:

        # Compute good data for each observation as % of total pixels
        print('Counting good quality pixels for each time step')
        data_perc = (pq_mask.sum(axis=[1, 2], dtype='int32') /
                     (pq_mask.shape[1] * pq_mask.shape[2]))

        # Filter by `min_gooddata` to drop low quality observations
        total_obs = len(ds.time)
        ds = ds.sel(time=data_perc >= min_gooddata)
        print(f'Filtering to {len(ds.time)} out of {total_obs} '
              f'time steps with at least {min_gooddata:.1%} '
              f'good quality pixels')
    
    # Drop bands not originally requested by user
    if requested_measurements:
        ds = ds[requested_measurements]
        
    ###############
    # Return data #
    ###############

    # Set nodata valuses using odc.algo tools to reduce peak memory
    # use when converting data to a float32 dtype
    ds = odc.algo.to_f32(ds)

    # If user supplied dask_chunks, return data as a dask array without
    # actually loading it in
    if dask_chunks is not None:
        print(f'Returning {len(ds.time)} time steps as a dask array')
        return ds
    else:
        print(f'Loading {len(ds.time)} time steps')
        return ds.compute()