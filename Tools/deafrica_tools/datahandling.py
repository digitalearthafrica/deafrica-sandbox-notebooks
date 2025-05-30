"""
Functions for loading and handling Digital Earth Africa data.
"""

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
from odc.algo import mask_cleanup
from copy import deepcopy
import odc.algo

from skimage.morphology import binary_erosion,binary_dilation,disk
from scipy.ndimage import binary_dilation
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from datetime import datetime
from dateutil import parser
from deafrica_tools.bandindices import calculate_indices

def _dc_query_only(**kw):
    """
    Remove load-only parameters, the rest
    can be passed to Query

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
    categories_to_mask_ls=dict(
        cloud="high_confidence", cloud_shadow="high_confidence"
    ),
    categories_to_mask_s2=[
        "cloud high probability",
        "cloud medium probability",
        "thin cirrus",
        "cloud shadows",
        "saturated or defective",
    ],
    categories_to_mask_s1=["invalid data"],
    mask_filters=None,
    mask_pixel_quality=True,
    ls7_slc_off=True,
    predicate=None,
    dtype="auto",
    verbose=True,
    **kwargs,
):
    """
    Loads analysis ready data.

    Loads and combines Landsat USGS Collections 2, Sentinel-2, and Sentinel-1 for
    multiple sensors (i.e. ls5t, ls7e, ls8c and ls9 for Landsat; s2a and s2b for Sentinel-2),
    optionally applies pixel quality masks, and drops time steps that
    contain greater than a minimum proportion of good quality (e.g. non-
    cloudy or shadowed) pixels.

    The function supports loading the following DE Africa products:

    Landsat:
        * ls5_sr ('sr' denotes surface reflectance)
        * ls7_sr
        * ls8_sr
        * ls9_sr
        * ls5_st ('st' denotes surface temperature)
        * ls7_st
        * ls8_st
        * ls9_st

    Sentinel-2:
        * s2_l2a
        * s2_l2a_c1

    Sentinel-1:
        * s1_rtc

    Last modified: Feb 2021

    Parameters
    ----------
    dc : datacube Datacube object
        The Datacube to connect to, i.e. `dc = datacube.Datacube()`.
        This allows you to also use development datacubes if required.
    products : list
        A list of product names to load data from. For example:

        * Landsat C2: ``['ls5_sr', 'ls7_sr', 'ls8_sr', 'ls9_sr']``
        * Sentinel-2: ``['s2_l2a', 's2_l2a_c1']``
        * Sentinel-1: ``['s1_rtc']``

    min_gooddata : float, optional
        An optional float giving the minimum percentage of good quality
        pixels required for a satellite observation to be loaded.
        Defaults to 0.0 which will return all observations regardless of
        pixel quality (set to e.g. 0.99 to return only observations with
        more than 99% good quality pixels).
    categories_to_mask_ls : dict, optional
        An optional dictionary that is used to identify poor quality pixels
        for masking. This mask is used for both masking out low
        quality pixels (e.g. cloud or shadow), and for dropping
        observations entirely based on the `min_gooddata` calculation.
    categories_to_mask_s2 : list, optional
        An optional list of Sentinel-2 Scene Classification Layer (SCL) names
        that identify poor quality pixels for masking.
    categories_to_mask_s1 : list, optional
        An optional list of Sentinel-1 mask names that identify poor
        quality pixels for masking.
    mask_filters : iterable of tuples, optional
        Iterable tuples of morphological operations - ("<operation>", <radius>)
        to apply on mask, where:

        operation: string, can be one of these morphological operations:
            * ``'closing'``  = remove small holes in cloud - morphological closing
            * ``'opening'``  = shrinks away small areas of the mask
            * ``'dilation'`` = adds padding to the mask
            * ``'erosion'``  = shrinks bright regions and enlarges dark regions

        radius: int
        e.g. ``mask_filters=[('erosion', 5),("opening", 2),("dilation", 2)]``
    mask_pixel_quality : bool, optional
        An optional boolean indicating whether to apply the poor data
        mask to all observations that were not filtered out for having
        less good quality pixels than ``min_gooddata``. E.g. if
        ``min_gooddata=0.99``, the filtered observations may still contain
        up to 1% poor quality pixels. The default of ``False`` simply
        returns the resulting observations without masking out these
        pixels; ``True`` masks them and sets them to NaN using the poor data
        mask. This will convert numeric values to floating point values
        which can cause memory issues, set to False to prevent this.
    ls7_slc_off : bool, optional
        An optional boolean indicating whether to include data from
        after the Landsat 7 SLC failure (i.e. SLC-off). Defaults to
        ``True``, which keeps all Landsat 7 observations > May 31 2003.
    predicate : function, optional
        An optional function that can be passed in to restrict the
        datasets that are loaded by the function. A filter function
        should take a `datacube.model.Dataset` object as an input (i.e.
        as returned from `dc.find_datasets`), and return a boolean.
        For example, a filter function could be used to return True on
        only datasets acquired in January:
        ``dataset.time.begin.month == 1``
    dtype : string, optional
        An optional parameter that controls the data type/dtype that
        layers are coerced to after loading. Valid values: ''`native`'',
        ``'auto'``, ``'float{16|32|64}'``.
        When ``'auto'`` is used, the data will be
        converted to ``'float32'`` if masking is used, otherwise data will
        be returned in the native data type of the data. Be aware that
        if data is loaded in its native dtype, nodata and masked
        pixels will be returned with the data's native nodata value
        (typically ``-999``), not ``NaN``.
        NOTE: If loading Landsat, the data is automatically rescaled so
        'native' dtype will return a value error.
    verbose : bool, optional
        If True, print progress statements during loading
    **kwargs : dict, optional
        A set of keyword arguments to ``dc.load`` that define the
        spatiotemporal query used to extract data. This typically
        includes ``measurements``, ``x`, ``y``, ``time``, ``resolution``,
        ``resampling``, ``group_by`` and ``crs``. Keyword arguments can
        either be listed directly in the ``load_ard`` call like any
        other parameter (e.g. ``measurements=['red']``), or by
        passing in a query kwarg dictionary (e.g. ``**query``). For a
        list of possible options, see the ``dc.load`` documentation:
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
    if verbose:
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
            "Please provide a list of product names to load data from. "
            "Valid options are: Landsat C2 SR: ['ls5_sr', 'ls7_sr', 'ls8_sr', 'ls9_sr'], or "
            "Landsat C2 ST: ['ls5_st', 'ls7_st', 'ls8_st', 'ls9_st'], or "
            "Sentinel-2: ['s2_l2a', 's2_l2a_c1'], or"
            "Sentinel-1: ['s1_rtc'], or"
        )
    
    # convert products to list if user passed as a string
    if type(products) == str:
        products=[products]
    
    if all(["ls" in product for product in products]):
        product_type = "ls"
    elif all(["s2" in product for product in products]):
        product_type = "s2"
    elif all(["s1" in product for product in products]):
        product_type = "s1"

    # check if the landsat product is surface temperature
    st = False
    if (product_type == "ls") & (all(["st" in product for product in products])):
        st = True

    # Check some parameters before proceeding
    if (product_type == "ls") & (dtype == "native"):
        raise ValueError(
            "Cannot load Landsat bands in native dtype "
            "as values require rescaling which converts dtype to float"
        )

    if product_type == "ls":
        if any(k in categories_to_mask_ls for k in ("cirrus", "cirrus_confidence")):
            raise ValueError(
                "'cirrus' categories for the pixel quality mask"
                " are not supported by load_ard"
            )

    # If `measurements` are specified but do not include pixel quality bands,
    #  add these to `measurements` according to collection
    if product_type == "ls":
        if verbose:
            print("Using pixel quality parameters for USGS Collection 2")
        fmask_band = "pixel_quality"

    elif product_type == "s2":
        if verbose:
            print("Using pixel quality parameters for Sentinel 2")
        fmask_band = "SCL"

    elif product_type == "s1":
        if verbose:
            print("Using pixel quality parameters for Sentinel 1")
        fmask_band = "mask"

    measurements = requested_measurements.copy() if requested_measurements else None

    # define a list of acceptable aliases to load landsat. We can't rely on 'common'
    # measurements as native band names have the same name for different measurements.
    ls_aliases = ["pixel_quality", "radiometric_saturation"]
    if st:
        ls_aliases = [
            "surface_temperature",
            "surface_temperature_quality",
            "atmospheric_transmittance",
            "thermal_radiance",
            "emissivity",
            "emissivity_stddev",
            "cloud_distance",
            "upwell_radiance",
            "downwell_radiance",
        ] + ls_aliases
    else:
        ls_aliases = ["red", "green", "blue", "nir", "swir_1", "swir_2"] + ls_aliases

    if measurements is not None:
        if product_type == "ls":

            # check we aren't loading aerosol bands from LS8
            aerosol_bands = [
                "aerosol_qa",
                "qa_aerosol",
                "atmos_opacity",
                "coastal_aerosol",
                "SR_QA_AEROSOL",
            ]
            if any(b in aerosol_bands for b in measurements):
                raise ValueError(
                    "load_ard doesn't support loading aerosol or "
                    "atmospeheric opacity related bands "
                    "for Landsat, instead use dc.load()"
                )

            # check measurements are in acceptable aliases list for landsat
            if set(measurements).issubset(ls_aliases):
                pass
            else:
                raise ValueError(
                    "load_ard does not support all band aliases for Landsat, "
                    "use only the following band names to load Landsat data: "
                    + str(ls_aliases)
                )

    # Deal with "load all" case: pick a set of bands common across
    # all products
    if measurements is None:
        if product_type == "ls":
            measurements = ls_aliases
        else:
            measurements = _common_bands(dc, products)

    # If `measurements` are specified but do not include pq, add.
    if measurements:
        if fmask_band not in measurements:
            measurements.append(fmask_band)

    # Get list of data and mask bands so that we can later exclude
    # mask bands from being masked themselves (also handle the case of rad_sat)
    data_bands = [
        band
        for band in measurements
        if band not in (fmask_band, "radiometric_saturation")
    ]
    mask_bands = [band for band in measurements if band not in data_bands]

    #################
    # Find datasets #
    #################

    # Pull out query params only to pass to dc.find_datasets
    query = _dc_query_only(**kwargs)

    # Extract datasets for each product using subset of dcload_kwargs
    dataset_list = []

    # Get list of datasets for each product
    if verbose:
        print("Finding datasets")
    for product in products:

        # Obtain list of datasets for product
        if verbose:
            print(f"    {product}")

        if product_type == "ls":
            # handle LS seperately to S2/S1 due to collection_category
            # force the user to load Tier 1
            datasets = dc.find_datasets(
                product=product, collection_category='T1', **query
            )
        else:
            datasets = dc.find_datasets(product=product, **query)

        # Remove Landsat 7 SLC-off observations if ls7_slc_off=False
        if not ls7_slc_off and product in ["ls7_sr"]:
            if verbose:
                print("    Ignoring SLC-off observations for ls7")
            datasets = [
                i
                for i in datasets
                if i.time.begin < datetime(2003, 5, 31, tzinfo=pytz.UTC)
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
    # of datasets prior to load (this now redundant as dc.load now supports
    # a predicate filter)
    if predicate:
        if verbose:
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

    # collection 2 USGS
    if product_type == "ls":
        mask, _ = masking.create_mask_value(
            ds[fmask_band].attrs["flags_definition"], **categories_to_mask_ls
        )
        
        pq_mask = (ds[fmask_band] & mask) != 0
        
        # only run if data bands are present 
        if len(data_bands) > 0: 
            
        # identify pixels that will become negative after rescaling (but not 0 values)
            invalid = (
                    ((ds[data_bands] < (-1.0 * -0.2 / 0.0000275)) & (ds[data_bands] > 0))
                    .to_array(dim="band")
                    .any(dim="band")
                    )

        #merge masks
        pq_mask = np.logical_or(pq_mask, pq_mask)

    # sentinel 2
    if product_type == "s2":
        pq_mask = odc.algo.enum_to_bool(mask=ds[fmask_band],
                                        categories=categories_to_mask_s2)
        
    # sentinel 1
    if product_type == "s1":
        pq_mask = odc.algo.enum_to_bool(mask=ds[fmask_band],
                                        categories=categories_to_mask_s1)

    # The good data percentage calculation has to load in all `fmask`
    # data, which can be slow. If the user has chosen no filtering
    # by using the default `min_gooddata = 0`, we can skip this step
    # completely to save processing time
    if min_gooddata > 0.0:

        # Compute good data for each observation as % of total pixels.
        # Inveerting the pq_mask for this because cloud=True in pq_mask
        # and we want to sum good pixels
        if verbose:
            print("Counting good quality pixels for each time step")
        data_perc = (~pq_mask).sum(axis=[1, 2], dtype="int32") / (
            pq_mask.shape[1] * pq_mask.shape[2]
        )

        keep = (data_perc >= min_gooddata).persist()

        # Filter by `min_gooddata` to drop low quality observations
        total_obs = len(ds.time)
        ds = ds.sel(time=keep)
        pq_mask = pq_mask.sel(time=keep)

        if verbose:
            print(
                f"Filtering to {len(ds.time)} out of {total_obs} "
                f"time steps with at least {min_gooddata:.1%} "
                f"good quality pixels"
            )

    # morpholigcal filtering on cloud masks
    if (mask_filters is not None) & (mask_pixel_quality):
        if verbose:
            print(f"Applying morphological filters to pq mask {mask_filters}")
        pq_mask = mask_cleanup(pq_mask, mask_filters=mask_filters)

    ###############
    # Apply masks #
    ###############

    # Generate good quality data mask
    mask = None
    if mask_pixel_quality:
        if verbose:
            print("Applying pixel quality/cloud mask")
        mask = pq_mask

    # Split into data/masks bands, as conversion to float and masking
    # should only be applied to data bands
    ds_data = ds[data_bands]
    ds_masks = ds[mask_bands]
    
    # Remove sentinel-2 pixels valued 1 (scene edges, terrain shadow)
    if product_type == "s2":
        valid_data_mask = (ds_data > 1).to_array(dim="band").all(dim="band")
        ds_data =  odc.algo.keep_good_only(ds_data, where=valid_data_mask)
        
    # Mask data if either of the above masks were generated
    if mask is not None:
        ds_data = odc.algo.erase_bad(ds_data, where=mask)
    
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

    # Apply the scale and offset factors to Collection 2 Landsat. We need
    # different factors for different bands. Also handle the case where
    # masking_pixel_quaity = False, in which case the dtype is still
    # in int, so we convert it to float
    if product_type == "ls":
        if verbose:
            print("Re-scaling Landsat C2 data")

        sr_bands = ["red", "green", "blue", "nir", "swir_1", "swir_2"]
        radiance_bands = ["thermal_radiance", "upwell_radiance", "downwell_radiance"]
        trans_emiss = ["atmospheric_transmittance", "emissivity", "emissivity_stddev"]
        qa = ["pixel_quality", "radiometric_saturation"]

        if mask_pixel_quality == False:
            # set nodata to NaNs before rescaling
            # in the case where masking hasn't already done this
            for band in ds.data_vars:
                if band not in qa:
                    ds[band] = odc.algo.to_f32(ds[band])

        for band in ds.data_vars:
            if band == "cloud_distance":
                ds[band] = 0.01 * ds[band]

            if band == "surface_temperature_quality":
                ds[band] = 0.01 * ds[band]

            if band in radiance_bands:
                ds[band] = 0.001 * ds[band]

            if band in trans_emiss:
                ds[band] = 0.0001 * ds[band]

            if band in sr_bands:
                ds[band] = 2.75e-5 * ds[band] - 0.2

            if band == "surface_temperature":
                ds[band] = ds[band] * 0.00341802 + 149.0

        # add back attrs that are lost during scaling calcs
        for band in ds.data_vars:
            ds[band].attrs.update(attrs)
    
    if product == "s2_l2a_c1":
        if verbose:
            print("Re-scaling Sentinel-2 C1 data")

        sr_bands = ["coastal", "red", "green", "blue", "rededge1", "rededge2", 
                    "rededge3", "nir", "nir08", "nir09", "swir16", "swir22"]
        
        if mask_pixel_quality == False:
            # set nodata to NaNs before rescaling
            # in the case where masking hasn't already done this
            for band in ds.data_vars:
                if band not in qa:
                    ds[band] = odc.algo.to_f32(ds[band])

        for band in ds.data_vars:
            if band in sr_bands:
                ds[band] = ds[band] - 1000
        
        # add back attrs that are lost during scaling calcs
        for band in ds.data_vars:
            ds[band].attrs.update(attrs)

    # If user supplied dask_chunks, return data as a dask array without
    # actually loading it in
    if dask_chunks is not None:
        if verbose:
            print(f"Returning {len(ds.time)} time steps as a dask array")
        return ds
    else:
        if verbose:
            print(f"Loading {len(ds.time)} time steps")
        return ds.compute()


def array_to_geotiff(
    fname, data, geo_transform, projection, nodata_val=0, dtype=gdal.GDT_Float32
):
    """
    Create a single band GeoTIFF file with data from an array.

    Because this works with simple arrays rather than xarray datasets
    from DEA, it requires geotransform info (`(upleft_x, x_size,
    x_rotation, upleft_y, y_rotation, y_size)`) and projection data
    (in "WKT" format) for the output raster. These are typically
    obtained from an existing raster using the following GDAL calls:

    >>> from osgeo import gdal
    >>> gdal_dataset = gdal.Open(raster_path)
    >>> geotrans = gdal_dataset.GetGeoTransform()
    >>> prj = gdal_dataset.GetProjection()

    or alternatively, directly from an xarray dataset:

    >>> geotrans = xarraydataset.geobox.transform.to_gdal()
    >>> prj = xarraydataset.geobox.crs.wkt


    Parameters
    ----------
    fname : str
        Output geotiff file path including extension
    data : numpy array
        Input array to export as a geotiff
    geo_transform : tuple
        Geotransform for output raster; e.g. `(upleft_x, x_size,
        x_rotation, upleft_y, y_rotation, y_size)`
    projection : str
        Projection for output raster (in "WKT" format)
    nodata_val : int, optional
        Value to convert to nodata in the output raster; default 0
    dtype : gdal dtype object, optional
        Optionally set the dtype of the output raster; can be
        useful when exporting an array of float or integer values.
        Defaults to `gdal.GDT_Float32`

    """

    # Set up driver
    driver = gdal.GetDriverByName("GTiff")

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
    str
        A EPSG string giving the most common CRS from all datasets returned
        by the query above

    """

    # remove dask_chunks & align to prevent func failing
    # prevent function altering dictionary kwargs
    query = deepcopy(query)
    if "dask_chunks" in query:
        query.pop("dask_chunks", None)

    if "align" in query:
        query.pop("align", None)

    # List of matching products
    matching_datasets = dc.find_datasets(product=product, **query)

    # Extract all CRSs
    crs_list = [str(i.crs) for i in matching_datasets]

    # Identify most common CRS
    crs_counts = Counter(crs_list)
    crs_mostcommon = crs_counts.most_common(1)[0][0]

    # Warn user if multiple CRSs are encountered
    if len(crs_counts.keys()) > 1:

        warnings.warn(
            f"Multiple UTM zones {list(crs_counts.keys())} "
            f"were returned for this query. Defaulting to "
            f"the most common zone: {crs_mostcommon}",
            UserWarning,
        )

    return crs_mostcommon


def download_unzip(url, output_dir=None, remove_zip=True):
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
    if not zip_name.endswith(".zip"):
        raise ValueError(
            f"The URL provided does not point to a .zip "
            f"file (e.g. {zip_name}). Please specify a "
            f"URL path to a valid .zip file"
        )

    # Download zip file
    print(f"Downloading {zip_name}")
    r = requests.get(url)
    with open(zip_name, "wb") as f:
        f.write(r.content)

    # Extract into output_dir
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(output_dir)
        print(
            f"Unzipping output files to: "
            f"{output_dir if output_dir else os.getcwd()}"
        )

    # Optionally cleanup
    if remove_zip:
        os.remove(zip_name)


def wofs_fuser(dest, src):
    """
    Fuse two WOfS water measurements represented as `ndarray` objects.

    Note: this is a copy of the function located here:
    https://github.com/GeoscienceAustralia/digitalearthau/blob/develop/digitalearthau/utils.py
    """
    empty = (dest & 1).astype(bool)
    both = ~empty & ~((src & 1).astype(bool))
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
    array
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

    return ~binary_dilation(
        array.astype(bool), structure=kernel.reshape((1,) + kernel.shape)
    )


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


def nearest(
    array: xr.DataArray, dim: str, target, index_name: str = None
) -> xr.DataArray:
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
        nearest_array[index_name] = xr.where(
            is_before_closer, da_before[index_name], da_after[index_name]
        )
    return nearest_array

def parallel_apply(ds, dim, func, *args):
    """
    Applies a custom function in parallel along the dimension of an
    xarray.Dataset or xarray.DataArray.

    The function can be any function that can be applied to an
    individual xarray.Dataset or xarray.DataArray (e.g. data for a
    single timestep). The function should also return data in
    xarray.Dataset or xarray.DataArray format.

    This function is useful as a simple method for parallising code
    that cannot easily be parallised using Dask.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        xarray data with a dimension `dim` to apply the custom function
        along.
    dim : string
        The dimension along which the custom function will be applied.
    func : function
        The function that will be applied in parallel to each array
        along dimension `dim`. The first argument passed to this
        function should be the array along `dim`.
    *args :
        Any number of arguments that will be passed to `func`.

    Returns
    -------
    xarray.Dataset
        A concatenated dataset containing an output for each array
        along the input `dim` dimension.
    """

    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    from itertools import repeat

    with ProcessPoolExecutor() as executor:

        # Apply func in parallel
        groups = [group.squeeze(dim=dim) for (i, group) in ds.groupby(dim)]
        to_iterate = (groups, *(repeat(i, len(groups)) for i in args))
        out_list = list(tqdm(executor.map(func, *to_iterate), total=len(groups)))

    # Combine to match the original dataset
    return xr.concat(out_list, dim=ds[dim])


def pan_sharpen_brovey(band_1, band_2, band_3, pan_band):
    """
    Brovey pan sharpening on surface reflectance input using numexpr
    and return three xarrays.
    Parameters
    ----------
    band_1, band_2, band_3 : xarray.DataArray or numpy.array
        Three input multispectral bands, either as xarray.DataArrays or
        numpy.arrays. These bands should have already been resampled to
        the spatial resolution of the panchromatic band.
    pan_band : xarray.DataArray or numpy.array
        A panchromatic band corresponding to the above multispectral
        bands that will be used to pan-sharpen the data.
    Returns
    -------
    band_1_sharpen, band_2_sharpen, band_3_sharpen : numpy.arrays
        Three numpy arrays equivelent to `band_1`, `band_2` and `band_3`
        pan-sharpened to the spatial resolution of `pan_band`.
    """
    # Calculate total
    exp = 'band_1 + band_2 + band_3'
    total = numexpr.evaluate(exp)

    # Perform Brovey Transform in form of: band/total*panchromatic
    exp = 'a/b*c'
    band_1_sharpen = numexpr.evaluate(exp, local_dict={'a': band_1,
                                                       'b': total,
                                                       'c': pan_band})
    band_2_sharpen = numexpr.evaluate(exp, local_dict={'a': band_2,
                                                       'b': total,
                                                       'c': pan_band})
    band_3_sharpen = numexpr.evaluate(exp, local_dict={'a': band_3,
                                                       'b': total,
                                                       'c': pan_band})

    return band_1_sharpen, band_2_sharpen, band_3_sharpen

def load_s1_by_orbits(dc,query):
    '''
    Function to query and load ascending and descending Sentinel-1 data 
    and add a variable to denote acquisition orbits
    
    Parameters:
    dc: connected datacube
    query: a query dictionary to define spatial extent, measurements, time range and spatial resolution
    
    Returns:
    Queried dataset with variable 'is_ascending' added to denote orbit path
    
    '''
    # load ascending data
    print('\nQuerying and loading Sentinel-1 ascending data...')
    ds_s1_ascending=load_ard(dc=dc,products=['s1_rtc'],resampling='bilinear',
                             dtype='native',sat_orbit_state='ascending',**query)
    # add an variable denoting data source
    ds_s1_ascending['is_ascending']=xr.DataArray(np.ones(len(ds_s1_ascending.time)),
                                                 dims=('time'),coords={'time': ds_s1_ascending.time})
    
    # load descending data
    print('\nQuerying and loading Sentinel-1 descending data...')
    ds_s1_descending=load_ard(dc=dc,products=['s1_rtc'],resampling='bilinear',
                              dtype='native',sat_orbit_state='descending',**query)
    # add an variable denoting data source
    ds_s1_descending['is_ascending']=xr.DataArray(np.zeros(len(ds_s1_descending.time)),
                                                  dims=('time'),coords={'time': ds_s1_descending.time})
    
    # merge datasets together
    ds_s1=xr.concat([ds_s1_ascending,ds_s1_descending],dim='time').sortby('time')
    
    return ds_s1

def filter_obs_by_orbit(ds_s1):
    '''
    Function to impliment per-pixel filtering of Sentinel-1 observations 
    to keep only observations from the orbit (ascending/descending) with higher frequency over time. 
    
    Each of the Sentinel-1 observations was acquired from either a descending or ascending orbit, 
    which has impacts on the local incidence angle and backscattering value. 
    Here we do the filtering to minimise the effects of inconsistent looking angle and obit direction for each individual pixel.

    Parameters:
    ds_s1: xarray.Dataset
        Time-series observations of Sentinel-1 data, 
        with two required variables: 'is_ascending' denoting orbit path and 'mask' to identify acquisition exent
    
    Returns:
    ds_s1_filtered: xarray.Dataset
        Filtered dataset
    '''

    print('\nFiltering Sentinel-1 product by orbit...')
    cnt_ascending=((ds_s1["is_ascending"]==1)&(ds_s1['mask']!=0)).sum(dim='time')
    cnt_descending=((ds_s1["is_ascending"]==0)&(ds_s1['mask']!=0)).sum(dim='time')
    
    ds_s1_filtered=ds_s1.where(((cnt_ascending>=cnt_descending)&(ds_s1["is_ascending"]==1))|
                               ((cnt_ascending<cnt_descending)&(ds_s1["is_ascending"]==0)))
    # remove intermediate variable
    ds_s1_filtered=ds_s1_filtered.drop_vars(["is_ascending"])
    # drop all-nan time steps
    ds_s1_filtered=ds_s1_filtered.dropna(dim='time',how='all')
    
    return ds_s1_filtered

def lee_filter(da, size):
    """
    Function to apply lee filter of specified window size.
    Adapted from https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
    
    Parameters:
    da: Xarray.dataArray, input single-time image data
    size: integer, filtering size in pixels
    
    Returns:
    numpy array of filtered image

    """
    img = da.values
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    
    return img_output

def preprocess_s1(ds_s1,filter_size=None,s1_orbit_filtering=True):
    '''
    Function to implement preprocessing on Sentinel-1 data, 
    including speckle filtering (optional), filtering observations by orbit (optional) and conversion to dB
    
    Parameters:
    ds_s1: xarray.Dataset
        Time-series of Sentinel-1 data, with variable 'vh' required
    filter_size: integer or None
        Speckle filtering size
    s1_orbit_filtering: Boolean
        Whether to filter Sentinel-1 observations by orbit
        
    Returns:
        xarray.Dataset
        Preprocessed Sentinel-1 data
    '''
    ds_s1_filtered=ds_s1
    
    # apply Lee filtering if required
    if not filter_size is None:
        print('Applying Lee filtering using filtering size of {} pixels...'.format(filter_size))
        # The lee filter above doesn't handle null values
        # We therefore set null values to 0 before applying the filter
        ds_s1_filtered = ds_s1.where(np.isfinite(ds_s1), 0)
        # Create a new entry in dataset corresponding to filtered VV and VH data
        ds_s1_filtered["vh"] = ds_s1_filtered.vh.groupby("time").apply(lee_filter, size=filter_size)
        # Null pixels should remain null, but also including pixels changed to 0 due to the filtering
        ds_s1_filtered['vh'] = ds_s1_filtered.vh.where(ds_s1_filtered.vh!=0,np.nan)
    
    # filter observations by orbit if required
    if s1_orbit_filtering:
        ds_s1_filtered=filter_obs_by_orbit(ds_s1_filtered)

    # Scale to plot data in decibels
    ds_s1_filtered['vh'] = 10 * np.log10(ds_s1_filtered.vh)
    return ds_s1_filtered

def get_mean_number_freq_valid_obs(da,mask,time_step):
    '''
    Calculate mean number of clear observations within each year/timestep in a masked zone
    
    Parameters:
    da: xarray.DataArray
        Time series of a single satellite band/variable (e.g. MNDWI)
    mask: xarray.DataArray or None
        A mask where the calculation is restricted within
    time_step: string
        Pre-defined time step for temporal aggregation, e.g. '1Y'
    
    Returns:
    n_valid_obs: xarray.DataArray
        Average number of valid observations within the coastal zone and for each time step
    freq_valid: xarray.DataArray
        Average frequency of valid observations within the coastal zone and for each time step
    '''
    n_valid_obs=(~da.isnull()).resample(time=time_step).sum('time').compute()
    freq_valid=(~da.isnull()).resample(time=time_step).mean('time').compute()
    if mask is None:
        n_valid_obs=n_valid_obs.mean(dim=['x','y'])
        freq_valid=freq_valid.mean(dim=['x','y'])
    else:
        n_valid_obs=n_valid_obs.where(mask).mean(dim=['x','y'])
        freq_valid=freq_valid.where(mask).mean(dim=['x','y'])
    
    return n_valid_obs,freq_valid

def create_coastal_mask(da,buffer_pixels):
    '''
    Create a simplified coastal zone mask based on time series of Sentinel-2 MNDWI data
    
    Parameters:
    ds_summaries: xarray.DataArray
        Time series of Sentinel-2 MNDWI data
    buffer_pixels: integer
        Number of pixels to buffer coastal zone
    
    Returns:
    coastal_mask: xarray.DataArray 
        A single time buffered coastal zone mask (0: non-coastal and 1: coastal)
    '''
    print('\nCalculating simplified coastal zone mask...')
    # apply thresholding and re-apply nodata values
    nodata = da.isnull()
    thresholded_ds = da>=0
    thresholded_ds = thresholded_ds.where(~nodata)
    # use 20% ~ 80% wet frequency to identify potential coastal zone
    coastal_mask=(thresholded_ds.mean(dim='time') >= 0.2)&(thresholded_ds.mean(dim='time') <= 0.8)
    # buffering
    print('\nApplying buffering of {} Sentinel-2 pixels (parameter buffer_pixels)...'.format(buffer_pixels))
    coastal_mask=xr.apply_ufunc(binary_dilation,coastal_mask.compute(),disk(buffer_pixels))
    return coastal_mask

def choose_product(ds_ls,ds_s2,ds_s1,ds_ls_s2,time_step,**kwargs):
    '''
    Rule-based guide on choosing the best availabel dataset in a given time step and optionally within a coastal zone mask
    
    Parameters:
    ds_ls: xarray.Dataset
        Time series Landsat data
    ds_s2: xarray.Dataset
        Time series Sentinel-2 data
    ds_s1: xarray.Dataset
        Time series Sentinel-1 data
    ds_ls_s2: xarray.Dataset or None
        Time series of combined Landsat and Sentinel-2 data.
    time_step: string
        Time step for temporal composition
    **kwargs: A set of optional parameters including:
        thresh_n_valid: integer
            Threhold of minimum average number of valid observations within each time step
        thresh_freq: float
            Threshold of minimum frequency of valid observations within each time step
        buffer_pixels: integer
            Number of pixels to buffer coastal zone
        coastal_masking: Boolean 
            whether to calculate a coastal zone mask and restrict the comparison of the products within the mask
    Returns:
        Xarray.Dataset of the best product
        String of the best product name: 'ls', 's2', 's1' or 'ls_s2'
    '''
    
    # check if optional parameters are defined otherwise set default values
    thresh_n_valid=10 if "thresh_n_valid" not in kwargs else kwargs["thresh_n_valid"]
    thresh_freq=0.2 if "thresh_freq" not in kwargs else kwargs["thresh_freq"]
    buffer_pixels=100 if "buffer_pixels" not in kwargs else kwargs["buffer_pixels"]
    print('\nThreshold number of valid observations (parameter thresh_n_valid): {}'.format(thresh_n_valid))
    print('\nThreshold frequency of valid observations (parameter thresh_freq): {}'.format(thresh_freq))
    
    # create mask if requested
    coastal_masking=False if "coastal_masking" not in kwargs else kwargs["coastal_masking"]
    if coastal_masking==True:
        # calculate index
        ds_s2 = calculate_indices(ds_s2, index='MNDWI', satellite_mission='s2')
        mask=create_coastal_mask(ds_s2['MNDWI'],buffer_pixels)
    else:
        print('\nNo coastal masking required, using all pixels within the selected region...')
        mask=None

    # calculate mean number and fraction of clear observations within each timestep and the mask
    print('\nCalculating number and frequency of valid observations...')
    
    n_valid_obs_s2,freq_valid_s2=get_mean_number_freq_valid_obs(ds_s2['green'],mask,time_step)
    print('\nSentinel-2: Average number and frequency of valid observations: {:.0f} and {:.2f}'.format(n_valid_obs_s2.mean().values,freq_valid_s2.mean().values))
    
    n_valid_obs_ls,freq_valid_ls=get_mean_number_freq_valid_obs(ds_ls['green'],mask,time_step)
    print('\nLandsat: Average number and frequency of valid observations: {:.0f} and {:.2f}'.format(n_valid_obs_ls.mean().values,freq_valid_ls.mean().values))
    
#     n_valid_obs_s1,freq_valid_s1=get_mean_number_freq_valid_obs(ds_s1['vh'],mask,time_step) # dont need this as sentinel-1 will only be chosen when optical datasets are not sufficient
    if not ds_ls_s2 is None:
        n_valid_obs_ls_s2,freq_valid_ls_s2=get_mean_number_freq_valid_obs(ds_ls_s2['green'],mask,time_step)
        print('\nCombined Landsat and Sentinel-2 product: Average number and frequency of valid observations: {:.0f} and {:.2f}'.format(n_valid_obs_ls_s2.mean().values,freq_valid_ls_s2.mean().values))
        
    # apply decision rules
    print('\nApplying rules to choose product...')
    
    # if Sentinel-2 meets requirements 
    if ((n_valid_obs_s2>=thresh_n_valid).all()) and ((freq_valid_s2>=thresh_freq).all()):
        print('\nSentinel-2 product has met the minimum required average number and frequency of valid observations within all time periods')
        # if combined product is available, choose combined product if it has both higher number and frequency
        if not ds_ls_s2 is None:
            if ((n_valid_obs_ls_s2>n_valid_obs_s2).all()) and ((freq_valid_ls_s2>freq_valid_s2).all()):
                ds_selected, product_name=ds_ls_s2,'ls_s2'
                print('\nChoosing combined Landsat and Sentinel-2 product as it has both higher number and frequency of valid observations within all time periods')
            else:
                ds_selected, product_name=ds_s2,'s2'
                print('\nChoosing Sentinel-2 product as neither Landsat or the combined product meets both requirements or is significantly better than Sentinel-2')
        # if combined product is unavailable, choose Landsat if it has both higher number and frequency
        elif ((n_valid_obs_ls>=n_valid_obs_s2).all()) and ((freq_valid_ls>=freq_valid_s2).all()):
            ds_selected, product_name=ds_ls,'ls'
            print('\nChoosing Landsat product as it has both higher average number and frequency of valid observations within all time periods')
        # otherwise choose Sentinel-2
        else:
            ds_selected, product_name=ds_s2,'s2'
            print('\nChoosing Sentinel-2 product as Landsat product does not meet both requirements or is not significantly better than Sentinel-2')
    # if Sentinel-2 doesn't meet both requirements,but Landsat does, either choose Landsat or combined product if available
    elif ((n_valid_obs_ls>=thresh_n_valid).all()) and ((freq_valid_ls>=thresh_freq).all()):
        print('\nSentinel-2 does not meet the minimum required average number and frequency of valid observations within all time periods, but Landsat does')
        if not ds_ls_s2 is None:
            ds_selected, product_name=ds_ls_s2,'ls_s2'
            print('\nChoosing combined Landsat and Sentinel-2 product as it has both higher number and frequency of valid observations within all time periods')
        else:
            ds_selected, product_name=ds_ls,'ls'
            print('\nChoosing Landsat product')
    # if neither Sentinel-2 or Landsat meet both requirements, choose combined product if it meets requirements
    elif not ds_ls_s2 is None:
        print('\nNeither Sentinel-2 or Landsat meets the minimum required average number and frequency of valid observations within all time periods')
        # but the combined product meet requirements
        if ((n_valid_obs_ls_s2>=thresh_n_valid).all()) and ((freq_valid_ls_s2>=thresh_freq).all()):
            ds_selected, product_name=ds_ls_s2,'ls_s2'
            print('\nChoosing combined Landsat and Sentinel-2 product as it meets the minimum required average number and frequency of valid observations within all time periods')
        else: 
            ds_selected, product_name=ds_s1,'s1'
            print('\nChoosing Sentinel-1 as no other products available that meet the requirements')
    # otherwise choose Sentinel-1
    else:
        print('\nNeither Sentinel-2 or Landsat meets the minimum required average number and frequency of valid observations within all time periods')
        ds_selected, product_name=ds_s1,'s1'
        print('\nChoosing Sentinel-1 product as no other products available that meet the requirements')
        
    print('\nBest available product: ',product_name)
    return ds_selected, product_name

def load_combined_ls_s2(dc,query):
    '''function to query and load combined Landsat and Sentinel-2 data
    
    Parameters:
    dc: connected datacube
    query: a query dictionary to define spatial extent, time range, measurements and spatial resolution for both datasets
    
    Returns:
    ds_combined: Combined data as xarray.Dataset
    '''
    print('Querying and loading combined Landsat and Sentinel-2 products...')
    # Load available Landsat data resampled to Sentinel-2 resolution
    ds_ls = load_ard(dc=dc, products=['ls8_sr', 'ls9_sr'],align=(10, 10),
                     resampling='bilinear',**query)

    # add an variable denoting data source (for future analysis)
    is_ls=xr.DataArray(np.ones(len(ds_ls.time)),dims=('time'),coords={'time': ds_ls.time})
    ds_ls['is_ls'] = is_ls

    # Load Sentinel-2 data
    ds_s2 = load_ard(dc=dc,products=['s2_l2a'],resampling='bilinear',
              align=(10, 10),mask_filters=[("opening", 2), ("dilation", 5)],**query)
    # add an variable denoting data source (for future analysis)
    is_ls=xr.DataArray(np.zeros(len(ds_s2.time)),dims=('time'),coords={'time': ds_s2.time})
    ds_s2['is_ls'] = is_ls

    # merge two datasets together
    ds_combined=xr.concat([ds_ls,ds_s2],dim='time').sortby('time')
    
    return ds_combined

def load_best_available_ds(dc, lat_range, lon_range, time_range, time_step, **kwargs):
    '''
    Function to query, load and compare different products, select and return the best available product
    
    Parameters:
    dc: connected datacube
    lat_range: range of latitudes in tuple or list
    lon_range: range of longitude in tuple or list
    time_range: range of time to query the data in tuple or list
    time_step: string, pre-defined time step for temporal aggregation, e.g. '1Y'
    **kwargs: A set of optional parameters on data query or comparison between products which may include:
    combine_ls_s2: A boolean value indicating whether to include merged/stacked Landsat and Sentinel-2 products as an option. Default to False.
    set_resolution: integer of spatial resolution in metres to query all products
    coastal_masking: A boolean value indicating whether to calculate a mask 
        and restrict the comparison of the products within the masked zone.
    set_product: Set this to only query and load a pre-selected product, 'ls','s2','ls_s2' or 's1'
        i.e. no other products will be queried or compared.
    thresh_n_valid: Threhold of minimum average number of valid observations within each time step, integer
    thresh_freq: Threshold of minimum frequency of valid observations within each time step, float between 0~1
    buffer_pixels: Number of pixels to buffer coastal zone, integer
        
    Returns:
    ds_selected: selected product as xarray.Dataset
    product_name: name of selected product in string format, i.e. 'ls','s2','ls_s2','s1'
    '''
    # parse input time range to accommodate queries before and after 2017
    min_time=min(parser.parse(time_range_i,default=datetime(1987,1,1,0,0)) 
                 for time_range_i in time_range)
    if min_time<datetime(2017,1,1,0,0):
        ls_only=True
    else:
        ls_only=False
        
    # set resolution for query based on optional user input
    if "set_resolution" not in kwargs:
        print('No resolution pre-set, using default resolutions for individual products...')
        resolution_ls=(-30,30)
        resolution_s2=(-10,10)
        resolution_s1=(-20,20)
    else:
        print('Using pre-set spatial resolution for all products...')
        resolution_ls=resolution_s2=resolution_s1=(kwargs["set_resolution"]*(-1),kwargs["set_resolution"])
        
    # create base query for all products
    query = {'x': lon_range,'y': lat_range,'time': time_range,
             'measurements': ['red', 'green', 'blue', 'swir_1'],
             'resolution': resolution_ls, 'group_by':'solar_day',
             'dask_chunks': {'time': 1}}
    
    # Identify the most common projection system in the input query 
    output_crs = mostcommon_crs(dc=dc, product='ls8_sr', query=query)
    
    # update base query
    query.update({'output_crs':output_crs,'min_gooddata':0.2})

    # check if product is pre-set by user
    set_product=None if not "set_product" in kwargs else kwargs["set_product"]
    if not set_product is None:
        product_name=set_product
        
    # check if allowing combining Landsat and Sentinel-2 as an option
    combine_ls_s2=False if not "combine_ls_s2" in kwargs else kwargs["combine_ls_s2"]
    
    # query and load specified products as user provided as possible
    if set_product=='ls':
        print('\nPre-selected product: Landsat')
        ds_selected=load_ard(dc=dc, products=['ls8_sr', 'ls9_sr'],
                             resampling='bilinear',**query)
    elif set_product=='s2':
        print('\nPre-selected product: Sentinel-2')
        if ls_only:
            raise ValueError("Querying date earlier than 2018, please change your pre-selected product as Landsat or query time range.")
        query.update({'resolution': resolution_s2})
        ds_selected= load_ard(dc=dc,products=['s2_l2a'],resampling='bilinear',
                              mask_filters=[("opening", 2), ("dilation", 5)],**query)
    elif set_product=='s1':
        print('\nPre-selected product: Sentinel-1')
        if ls_only:
            raise ValueError("Querying date earlier than 2018, please change your pre-selected product as Landsat or query time range.")
        query.update({'resolution': resolution_s1,'measurements': ['vh','mask']})
        ds_selected=load_s1_by_orbits(dc,query)
    elif set_product=='ls_s2':
        print('\nPre-selected product: combined Landsat and Sentinel-2 products')
        if ("combine_ls_s2" in kwargs)and(combine_ls_s2==False):
            raise ValueError("Conflicting: requesting querying combination of Landsat and Sentinel-2 products while parameter combine_ls_s2 is disabled. Please change parameter and try to run the function again.")
        else:
            query.update({'resolution': resolution_s2})
            ds_selected=load_combined_ls_s2(dc,query)
    else: # no preselection of product or wrong input of product name
        print('\nNo pre-selection of products, querying and compare all products...')
        if ls_only:
            print('\nQuerying date earlier than 2018, only Landsat data will be queried and loaded.')
            # Load available Landsat data
            ds_ls = load_ard(dc=dc, products=['ls8_sr', 'ls9_sr'],
                             resampling='bilinear',**query)
            ds_selected=ds_ls
            product_name='ls'
        else:
            if combine_ls_s2==True:
                query.update({'resolution': resolution_s2})
                ds_ls_s2=load_combined_ls_s2(dc,query)
            else:
                print('\nCombined Landsat and Sentinel-2 products excluded from comparison')
                ds_ls_s2=None
                
            # Load available Landsat data
            print('\nQuerying and loading Landsat data...')
            ds_ls = load_ard(dc=dc, products=['ls8_sr', 'ls9_sr'],
                             resampling='bilinear',**query)
            
            # Load Sentinel-2 data
            print('\nQuerying and Sentinel-2 data...')
            query.update({'resolution': resolution_s2})
            ds_s2 = load_ard(dc=dc,products=['s2_l2a'],resampling='bilinear',
                      mask_filters=[("opening", 2), ("dilation", 5)],**query)
    
            # query and filter Sentinel-1 data by orbit
            query.update({'resolution': resolution_s1,'measurements': ['vh','mask']})
            ds_s1=load_s1_by_orbits(dc,query)
            # apply rules to choose best product
            ds_selected,product_name=choose_product(ds_ls,ds_s2,ds_s1,ds_ls_s2,time_step,**kwargs)
    
    # drop all-nan time steps
    ds_selected=ds_selected.dropna(dim='time',how='all')
    
    return ds_selected,product_name