from dc_mosaic import (ls7_unpack_qa, ls8_unpack_qa, ls5_unpack_qa)
import numpy as np
import xarray as xr

## Utils ##

def xarray_values_in(data, values, data_vars=None):
    """
    Returns a mask for an xarray Dataset or DataArray, with `True` wherever the value is in values.

    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray
        The data to check for value matches.
    values: list-like
        The values to check for.
    data_vars: list-like
        The names of the data variables to check.

    Returns
    -------
    mask: np.ndarray
        A NumPy array shaped like ``data``. The mask can be used to mask ``data``.
        That is, ``data.where(mask)`` is an intended use.
    """
    data_vars_to_check = data_vars if data_vars is not None else list(data.data_vars.keys())
    if isinstance(data, xr.Dataset):
        mask = np.full_like(data[data_vars_to_check[0]].values, False, dtype=np.bool)
        for data_arr in data[data_vars_to_check].values():
            for value in values:
                mask = mask | (data_arr.values == value)
    elif isinstance(data, xr.DataArray):
        mask = np.full_like(data, False, dtype=np.bool)
        for value in values:
            mask = mask | (data.values == value)
    return mask

## End Utils ##

## Misc ##

def create_2D_mosaic_clean_mask(clean_mask):
    """
    The clean mask of a mosaic should be determined by the compositing function (e.g. mean 
    mosaic, median mosaic, etc.). This is simply supposed to be a decent approximation of a 
    clean mask for a mosaic that has no time dimension.
    
    Parameters
    ----------
    clean_mask: np.ndarray
        The 3D clean mask used to construct the mosaic.
    
    Returns
    -------
    mosaic_clean_mask: np.ndarray
        A 2D clean mask for a mosaic.
    """
    mosaic_clean_mask = clean_mask[0]
    # Take the logical OR of clean masks through time.
    for i in range(1, clean_mask.shape[0]):
        mosaic_clean_mask = np.logical_or(mosaic_clean_mask, clean_mask[i])    
    return mosaic_clean_mask

def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a NumPy array mask with a circle.
    Credit goes to https://stackoverflow.com/a/44874588/5449970.

    Parameters
    ----------
    h, w: int
        The height and width of the data to mask, respectively.
    center: 2-tuple of int
        The center of the circle, specified as a 2-tuple of the x and y indices.
        By default, the center will be the center of the image.
    radius: numeric
        The radius of the circle.
        Be default, the radius will be the smallest distance between
        the center and the image walls.

    Returns
    -------
    mask: np.ndarray
        A boolean 2D NumPy array.
    """
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

## End Misc ##

## Landsat ##

def landsat_clean_mask_invalid(dataset):
    """
    Masks out invalid data according to the LANDSAT
    surface reflectance specifications. See this document:
    https://landsat.usgs.gov/sites/default/files/documents/ledaps_product_guide.pdf pages 19-20.

    Parameters
    ----------
    dataset: xarray.Dataset
        An `xarray.Dataset` containing bands such as 'red', 'green', or 'blue'.

    Returns
    -------
    invalid_mask: xarray.DataArray
        An `xarray.DataArray` with the same number and order of coordinates as in `dataset`.
        The `True` values specify what pixels are valid.
    """
    invalid_mask = None
    data_arr_names = [arr_name for arr_name in list(dataset.data_vars)
                      if arr_name not in ['pixel_qa', 'radsat_qa', 'cloud_qa']]
    # Only keep data where all bands are in the valid range.
    for i, data_arr_name in enumerate(data_arr_names):
        invalid_mask_arr = (0 < dataset[data_arr_name]) & (dataset[data_arr_name] < 10000)
        invalid_mask = invalid_mask_arr if i == 0 else (invalid_mask & invalid_mask_arr)
    return invalid_mask


def landsat_qa_clean_mask(dataset, platform, cover_types=['clear', 'water']):
    """
    Returns a clean_mask for `dataset` that masks out various types of terrain cover using the
    Landsat pixel_qa band. Note that Landsat masks specify what to keep, not what to remove.
    This means that using `cover_types=['clear', 'water']` should keep only clear land and water.

    See "pixel_qa band" here: https://landsat.usgs.gov/landsat-surface-reflectance-quality-assessment
    and Section 7 here: https://landsat.usgs.gov/sites/default/files/documents/lasrc_product_guide.pdf.

    Parameters
    ----------
    dataset: xarray.Dataset
        An xarray (usually produced by `datacube.load()`) that contains a `pixel_qa` data
        variable.
    platform: str
        A string denoting the platform to be used. Can be "LANDSAT_5", "LANDSAT_7", or
        "LANDSAT_8".
    cover_types: list
        A list of the cover types to include. Adding a cover type allows it to remain in the masked data.
        Cover types for all Landsat platforms include:
        ['fill', 'clear', 'water', 'shadow', 'snow', 'cloud', 'low_conf_cl', 'med_conf_cl', 'high_conf_cl'].

        'fill' removes "no_data" values, which indicates an absense of data. This value is -9999 for Landsat platforms.
        Generally, don't use 'fill'.
        'clear' allows only clear terrain. 'water' allows only water. 'shadow' allows only cloud shadows.
        'snow' allows only snow. 'cloud' allows only clouds, but note that it often only selects cloud boundaries.
        'low_conf_cl', 'med_conf_cl', and 'high_conf_cl' denote low, medium, and high confidence in cloud coverage.
        'low_conf_cl' is useful on its own for only removing clouds, however, 'clear' is usually better suited for this.
        'med_conf_cl' is useful in combination with 'low_conf_cl' to allow slightly heavier cloud coverage.
        Note that 'med_conf_cl' and 'cloud' are very similar.
        'high_conf_cl' is useful in combination with both 'low_conf_cl' and 'med_conf_cl'.

        For Landsat 8, there are more cover types: ['low_conf_cir', 'high_conf_cir', 'terrain_occ'].
        'low_conf_cir' and 'high_conf_cir' denote low and high confidence in cirrus clouds.
        'terrain_occ' allows only occluded terrain.

    Returns
    -------
    clean_mask: xarray.DataArray
        An xarray DataArray with the same number and order of coordinates as in `dataset`.
    """
    processing_options = {
        "LANDSAT_5": ls5_unpack_qa,
        "LANDSAT_7": ls7_unpack_qa,
        "LANDSAT_8": ls8_unpack_qa
    }

    clean_mask = None
    # Keep all specified cover types (e.g. 'clear', 'water'), so logically or the separate masks.
    for i, cover_type in enumerate(cover_types):
        cover_type_clean_mask = processing_options[platform](dataset.pixel_qa, cover_type)
        clean_mask = cover_type_clean_mask if i == 0 else (clean_mask | cover_type_clean_mask)
    return clean_mask

## End Landsat ##

## Sentinel 2 ##

def sentinel2_fmask_clean_mask(dataset, cover_types=['valid', 'water']):
    """
    Returns a clean_mask for `dataset` that masks out various types of terrain cover using the
    Sentinel 2 fmask band. Note that clean masks specify what to keep, not what to remove.
    This means that using `cover_types=['valid', 'water']` should keep only clear land and water.

    See "Classification Mask Generation" here:
    https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm

    Parameters
    ----------
    dataset: xarray.Dataset
        An xarray (usually produced by `datacube.load()`) that contains a `fmask` data
        variable.
    cover_types: list
        A list of the cover types to include. Adding a cover type allows it to remain in the masked data.
        Cover types for all Landsat platforms include:
        ['null', 'valid', 'cloud', 'cloud_shadow', 'snow', 'water'].

        'null' removes null values, which indicates an absense of data.
        'valid' allows clear views that are not cloud shadow, snow, or water.
        'cloud' allows clouds.
        'cloud_shadow' allows only cloud shadows.
        'snow' allows only snow.
        'water' allows only water.

        Here is a table of fmask values and their significances:
        Value Description
        0     Null
        1     Valid
        2     Cloud
        3     Cloud shadow
        4     Snow
        5     water

    Returns
    -------
    clean_mask: xarray.DataArray of boolean
        A boolean `xarray.DataArray` denoting which elements in `dataset` to keep -
        with the same number and order of coordinates as in `dataset`.
    """
    fmask_table = {'null': 0, 'valid': 1, 'cloud': 2, 'cloud_shadow': 3, 'snow': 4, 'water': 5}
    fmask_values_to_keep = [fmask_table[cover_type] for cover_type in cover_types]
    clean_mask = xarray_values_in(dataset.fmask, fmask_values_to_keep)
    return clean_mask

## End Sentinel 2 ##