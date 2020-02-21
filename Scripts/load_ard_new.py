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

def _split_dc_params(**kw):
    """ Partition parameters meant for `dc.load(..)` into query-time and load-time.
    Note that some parameters are used for both.
    Returns
    =======
    (query: dict, load: dict)
    """
    _nothing = object()

    def _impl(measurements=_nothing,
              output_crs=_nothing,
              resolution=_nothing,
              resampling=_nothing,
              skip_broken_datasets=_nothing,
              dask_chunks=_nothing,
              like=_nothing,
              fuse_func=_nothing,
              align=_nothing,
              datasets=_nothing,
              progress_cbk=_nothing,
              **query):
        if like is not _nothing:
            query = dict(like=like, **query)

        load_args = {k: v for k, v in locals().items() if v is not _nothing}
        load_args.pop('query')

        return query, load_args

    return _impl(**kw)


def load_ard(dc,
             products=None,
             min_gooddata=0.0,
             fmask_categories=['valid', 'snow', 'water'],
             mask_pixel_quality=True,
#              mask_contiguity='nbart_contiguity',
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
    fmask_categories : list, optional
        An optional list of fmask category names to treat as good
        quality observations in the above `min_gooddata` calculation.
        The default is `['valid', 'snow', 'water']` which will return
        non-cloudy or shadowed land, snow and water pixels. Choose from:
        'nodata', 'valid', 'cloud', 'shadow', 'snow', and 'water'.
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

    query, load_params = _split_dc_params(**extras)

    # We deal with `dask_chunks` separately
    dask_chunks = load_params.pop('dask_chunks', None)

    # Warn user if they combine lazy load with min_gooddata
    if (min_gooddata > 0.0) and dask_chunks is not None:
        warnings.warn("Setting 'min_gooddata' percentage to > 0.0 "
                      "will cause dask arrays to compute when "
                      "loading pixel-quality data to calculate "
                      "'good pixel' percentage. This can "
                      "slow the return of your dataset.")

    # Verify that products were provided, and that only Sentinel-2 or
    # only Landsat products are being loaded at the same time
    if not products:
        raise ValueError("Please provide a list of product names "
                         "to load data from. Valid options are: \n"
                         "['ls5_usgs_sr_scene', 'ls7_usgs_sr_scene', 'ls8_usgs_sr_scene'] "
                         "or ['usgs_ls5t_level2_2', 'usgs_ls7e_level2_2', 'usgs_ls8c_level2_2']
                         "for Landsat. Sentinel 2: ['s2a_msil2a', 's2b_msil2a']"
                         
    elif all(['ls' in product for product in products]):
        product_type = 'ls'
    elif all(['s2' in product for product in products]):
        product_type = 's2'
    
    # List of valid USGS Collection 1 products
    c1_products = ['ls5_usgs_sr_scene',
                   'ls7_usgs_sr_scene',
                   'ls8_usgs_sr_scene']

    # List of valid USGS Collection 2 products
    c2_products = ['usgs_ls5t_level2_2', 
                   'usgs_ls7e_level2_2', 
                   'usgs_ls8c_level2_2']
    
    # List of valid Sentinel 2 products
    s2_products = ['s2a_msil2a', 's2b_msil2a']
                         
    # If `measurements` are specified but do not include pixel quality bands,
    #  add these to `measurements` according to collection
    if product in c2_products:
        print('    Using pixel quality parameters for USGS Collection 2')
        fmask_band = 'quality_l2_aerosol'
                        
    elif product in c1_products:
        print('    Using pixel quality parameters for USGS Collection 1')
        fmask_band = 'pixel_qa'
    
    elif product in s2_products:
        print('    Using pixel quality parameters for Sentinel 2')
        fmask_band = 'scl'
    
    requested_measurements = load_params.pop('measurements', None)
    measurements = requested_measurements.copy() if requested_measurements else None
                     
    if measurements:
        if fmask_band not in measurements:
            measurements.append(fmask_band)

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
                 dask_chunks={} if dask_chunks is None else dask_chunks,
                 **load_params)

    ###############
    # Apply masks #
    ###############
    #need to distinguish between products due to different "fmask" band properties                     
    if product in c2_products:

                        
    elif product in c1_products:
        fmask_categories = {'cloud': 'no_cloud',
                            'cloud_shadow': 'no_cloud_shadow',
                            'nodata': False}
        pq_mask = masking.make_mask(ds[fmask_band], 
                                    **quality_flags_prod)
                         
    elif product in s2_products:
        fmask_categories = ['vegetation','snow or ice','water',
                            'bare soils','unclassified']
        pq_mask = odc.algo.fmask_to_bool(ds[fmask_band],
                                     categories=fmask_categories)

                         
                         
                         
                         
    # Calculate pixel quality mask
    pq_mask = odc.algo.fmask_to_bool(ds[fmask_band],
                                     categories=fmask_categories)

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