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
    """Remove load-only parameters, the rest can be passed to Query

    Returns
    =======

    dict of query parameters
    """

    def _impl(
        measurements=None,
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
        **query,
    ):
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


def load_ard(
    dc,
    products=None,
    min_gooddata=0.0,
    pq_categories_s2=[
        "vegetation",
        "snow or ice",
        "water",
        "bare soils",
        "unclassified",
        "dark area pixels",
    ],
    pq_categories_ls=None,
    mask_pixel_quality=True,
    ls7_slc_off=True,
    predicate=None,
    dtype="auto",
    scaling="raw",
    **kwargs,
):
    """
    Loads analysis ready data.

    Loads and combines Landsat Collections 1 or 2, and Sentinel-2 for
    multiple sensors (i.e. ls5t, ls7e and ls8c for Landsat; s2a and s2b for Sentinel-2),
    optionally applies pixel quality masks, and drops time steps that
    contain greater than a minimum proportion of good quality (e.g. non-
    cloudy or shadowed) pixels.

    The function supports loading the following DE Africa products:

        * ls5_c2l2
        * ls7_c2l2
        * ls8_c2l2
        * s2_l2a

    Last modified: March 2021

    Parameters
    ----------
    dc : datacube Datacube object
        The Datacube to connect to, i.e. `dc = datacube.Datacube()`.
        This allows you to also use development datacubes if required.
    products : list
        A list of product names to load data from. Valid options are:

        * Landsat C1: `['ls5_c2l2', 'ls7_c2l2', 'ls8_c2l2']`
        * Sentinel-2: `['s2_l2a']`

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
        pixel mask from the selected USGS product's pixel quality band.
        This mask is used for both masking out low
        quality pixels (e.g. cloud or shadow), and for dropping
        observations entirely based on the above `min_gooddata`
        calculation. Default is None, which will apply the following masks:

        for USGS Collection 2:

        >>> {'cloud_shadow': 'not_cloud_shadow',
        ...  'cloud_or_cirrus': 'not_cloud_or_cirrus',
        ...  'nodata': False}

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
        surface reflectance values have an offset so normalised band indices
        will return non-sensical results if setting scaling='raw'.
    **kwargs :
        A set of keyword arguments to `dc.load` that define the
        spatiotemporal query used to extract data. This typically
        includes `measurements`, `x`, `y`, `time`, `resolution`,
        `resampling`, `group_by` and `crs`. Keyword arguments can
        either be listed directly in the `load_ard` call like any
        other parameter (e.g. `measurements=['red']`), or by
        passing in a query kwarg dictionary (e.g. `**query`). For a
        list of possible options, see the `dc.load` documentation:
        https://datacube-core.readthedocs.io/en/latest/dev/api/generate/datacube.Datacube.load.html

    Returns
    -------
    combined_ds : xarray Dataset
        An xarray dataset containing only satellite observations that
        contains greater than `min_gooddata` proportion of good quality
        pixels.

    """

    #########
    # Setup #
    #########
    # prevent function altering original query object
    kwargs = deepcopy(kwargs)

    # We deal with `dask_chunks` separately
    dask_chunks = kwargs.pop("dask_chunks", None)
    requested_measurements = kwargs.pop("measurements", None)

    # Warn user if they combine lazy load with min_gooddata
    if (min_gooddata > 0.0) and dask_chunks is not None:
        warnings.warn(
            "Setting 'min_gooddata' percentage to > 0.0 "
            "will cause dask arrays to compute when "
            "loading pixel-quality data to calculate "
            "'good pixel' percentage. This can "
            "slow the return of your dataset."
        )

    # Verify that products were provided and determine if Sentinel-2
    # or Landsat data is being loaded
    if not products:
        raise ValueError(
            f"Please provide a list of product names " f"to load data from."
        )

    elif all(["ls" in product for product in products]):
        product_type = "ls"
    elif all(["s2" in product for product in products]):
        product_type = "s2"

    # If `measurements` are specified but do not include pixel quality bands,
    #  add these to `measurements` according to collection
    if (product_type == "ls"):
        print("Using pixel quality parameters for USGS Collection 2")
        fmask_band = "quality_pixel"

    elif product_type == "s2":
        print("Using pixel quality parameters for Sentinel 2")
        fmask_band = "SCL"

    measurements = requested_measurements.copy() if requested_measurements else None

    # Deal with "load all" case: pick a set of bands common across
    # all products
    if measurements is None:
        measurements = _common_bands(dc, products)

    # If `measurements` are specified but do not include pq, add.
    if measurements:
        if fmask_band not in measurements:
            measurements.append(fmask_band)

    # Get list of data and mask bands so that we can later exclude
    # mask bands from being masked themselves
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
    print("Finding datasets")
    for product in products:

        # Obtain list of datasets for product
        print(f"    {product}")
        datasets = dc.find_datasets(product=product, **query)

        # Remove Landsat 7 SLC-off observations if ls7_slc_off=False
        if not ls7_slc_off and product in ["ls7_c2l2"]:
            print("    Ignoring SLC-off observations for ls7")
            datasets = [
                i
                for i in datasets
                if i.time.begin < datetime.datetime(2003, 5, 31, tzinfo=pytz.UTC)
            ]

        # Add any returned datasets to list
        dataset_list.extend(datasets)

    # Raise exception if no datasets are returned
    if len(dataset_list) == 0:
        raise ValueError(
            "No data available for query: ensure that "
            "the products specified have data for the "
            "time and location requested"
        )

    # If predicate is specified, use this function to filter the list
    # of datasets prior to load
    if predicate:
        print(f"Filtering datasets using filter function")
        dataset_list = [ds for ds in dataset_list if predicate(ds)]

    # Raise exception if filtering removes all datasets
    if len(dataset_list) == 0:
        raise ValueError("No data available after filtering with " "filter function")

    #############
    # Load data #
    #############
    # Note we always load using dask here so that
    # we can lazy load data before filtering by good data
    ds = dc.load(
        datasets=dataset_list,
        measurements=measurements,
        dask_chunks={} if dask_chunks is None else dask_chunks,
        **kwargs,
    )

    ####################
    # Filter good data #
    ####################

    # need to distinguish between products due to different
    # pq band properties

    # collection 2 USGS or FC
    if product_type == "ls":
        if pq_categories_ls is None:
            quality_flags_prod = {
                "clear": True,
                "cloud_shadow": "not_high_confidence",
                "nodata": False
            }
        else:
            quality_flags_prod = pq_categories_ls

        pq_mask = masking.make_mask(ds[fmask_band], **quality_flags_prod)

    # sentinel 2
    if product_type == "s2":
        # currently broken for mask band values >=8
        # pq_mask = odc.algo.fmask_to_bool(ds[fmask_band],
        #                             categories=pq_categories_s2)
        flags_s2 = (
            dc.list_measurements()
            .loc[products[0]]
            .loc[fmask_band]["flags_definition"]["qa"]["values"]
        )
        pq_mask = ds[fmask_band].isin(
            [int(k) for k, v in flags_s2.items() if v in pq_categories_s2]
        )

    # The good data percentage calculation has to load in all `fmask`
    # data, which can be slow. If the user has chosen no filtering
    # by using the default `min_gooddata = 0`, we can skip this step
    # completely to save processing time
    if min_gooddata > 0.0:

        # Compute good data for each observation as % of total pixels
        print("Counting good quality pixels for each time step")
        data_perc = pq_mask.sum(axis=[1, 2], dtype="int32") / (
            pq_mask.shape[1] * pq_mask.shape[2]
        )

        keep = (data_perc >= min_gooddata).persist()

        # Filter by `min_gooddata` to drop low quality observations
        total_obs = len(ds.time)
        ds = ds.sel(time=keep)
        pq_mask = pq_mask.sel(time=keep)
        print(
            f"Filtering to {len(ds.time)} out of {total_obs} "
            f"time steps with at least {min_gooddata:.1%} "
            f"good quality pixels"
        )

    ###############
    # Apply masks #
    ###############

    # Generate good quality data mask
    mask = None
    if mask_pixel_quality:
        print("Applying pixel quality/cloud mask")
        mask = pq_mask

    # Split into data/masks bands, as conversion to float and masking
    # should only be applied to data bands
    ds_data = ds[data_bands]
    ds_masks = ds[mask_bands]

    # Mask data if either of the above masks were generated
    if mask is not None:
        ds_data = odc.algo.keep_good_only(ds_data, where=mask)

    # Automatically set dtype to either native or float32 depending
    # on whether masking was requested
    if dtype == "auto":
        dtype = "native" if mask is None else "float32"

    # Set nodata values using odc.algo tools to reduce peak memory
    # use when converting data dtype
    if dtype != "native":
        ds_data = odc.algo.to_float(ds_data, dtype=dtype)

    # Put data and mask bands back together
    attrs = ds.attrs
    ds = xr.merge([ds_data, ds_masks])
    ds.attrs.update(attrs)

    ###############
    # Return data #
    ###############

    # Drop bands not originally requested by user
    if requested_measurements:
        ds = ds[requested_measurements]

    # Scale data Sentinel-2 data 0-1 if requested
    if scaling == "normalised":
        if product_type == "s2":
            print("Re-scaling Sentinel-2 data")
            not_sr_bands = [
                "scl",
                "qa",
                "mask",
                "water_vapour",
                "aerosol_optical_thickness",
            ]
            
            for band in ds.data_vars:
                if band not in not_sr_bands:
                    ds[band] = ds[band] / 10000

    # Collection 2 Landsat raw values aren't useful so always rescale,
    # need different factors for surface-temp and SR
    if product_type == "ls":
        print("Re-scaling Landsat C2 data")
        not_sr_bands = [
            "thermal_radiance",
            "upwell_radiance",
            "upwell_radiance",
            "atmospheric_transmittance",
            "emissivity",
            "emissivity_stdev",
            "cloud_distance",
            "quality_aerosol",
            "quality_surface_temperature",
            "quality_pixel",
            "surface_temperature",
            "quality_radiometric_saturation",
            "quality_surface_temperature"
        ]

        for band in ds.data_vars:
            if band == "surface_temperature":
                ds[band] = ds[band] * 0.00341802 + 149.0

            if band not in not_sr_bands:
                ds[band] =  2.75e-5 * ds[band] - 0.2

    # If user supplied dask_chunks, return data as a dask array without
    # actually loading it in
    if dask_chunks is not None:
        print(f"Returning {len(ds.time)} time steps as a dask array")
        return ds
    else:
        print(f"Loading {len(ds.time)} time steps")
        return ds.compute()
