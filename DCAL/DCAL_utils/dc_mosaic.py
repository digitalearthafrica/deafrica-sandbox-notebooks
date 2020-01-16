# Copyright 2016 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. All Rights Reserved.
#
# Portion of this code is Copyright Geoscience Australia, Licensed under the
# Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License
# at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# The CEOS 2 platform is licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import numpy as np
import xarray as xr
from collections import OrderedDict
#import hdmedians as hd

import dc_utilities as utilities
from dc_utilities import create_default_clean_mask

"""
Utility Functions
"""

def convert_to_dtype(data, dtype):
    """
    A utility function converting xarray, pandas, or NumPy data to a given dtype.

    Parameters
    ----------
    data: xarray.Dataset, xarray.DataArray, pandas.Series, pandas.DataFrame,
             or numpy.ndarray
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.
    """
    if dtype is None: # Don't convert the data type.
        return data
    return data.astype(dtype)


"""
Compositing Functions
"""

def create_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, intermediate_product=None, **kwargs):
    """
    Creates a most-recent-to-oldest mosaic of the input dataset.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, latitude, longitude
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: latitude, longitude
        variables: same as dataset_in
    """
    dataset_in = dataset_in.copy(deep=True)

    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    # Mask data with clean_mask. All values where clean_mask==False are set to no_data.
    for key in list(dataset_in.data_vars):
        dataset_in[key].values[np.invert(clean_mask)] = no_data

    dataset_in_dtypes = None
    if dtype is None:
        # Save dtypes because masking with Dataset.where() converts to float64.
        band_list = list(dataset_in.data_vars)
        dataset_in_dtypes = {}
        for band in band_list:
            dataset_in_dtypes[band] = dataset_in[band].dtype

    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None

    time_slices = reversed(
        range(len(dataset_in.time))) if 'reverse_time' in kwargs else range(len(dataset_in.time))
    for index in time_slices:
        dataset_slice = dataset_in.isel(time=index).drop('time')
        if dataset_out is None:
            dataset_out = dataset_slice.copy(deep=True)
            utilities.clear_attrs(dataset_out)
        else:
            for key in list(dataset_in.data_vars):
                dataset_out[key].values[dataset_out[key].values == -9999] = dataset_slice[key].values[dataset_out[key]
                                                                                                      .values == -9999]
                dataset_out[key].attrs = OrderedDict()

    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out

def create_mean_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, **kwargs):
    """
    Method for calculating the mean pixel value for a given dataset.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, latitude, longitude
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: latitude, longitude
        variables: same as dataset_in
    """
    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    dataset_in_dtypes = None
    if dtype is None:
        # Save dtypes because masking with Dataset.where() converts to float64.
        band_list = list(dataset_in.data_vars)
        dataset_in_dtypes = {}
        for band in band_list:
            dataset_in_dtypes[band] = dataset_in[band].dtype

    # Mask out clouds and scan lines.
    dataset_in = dataset_in.where((dataset_in != no_data) & (clean_mask))
    dataset_out = dataset_in.mean(dim='time', skipna=True, keep_attrs=False)

    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out


def create_median_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, **kwargs):
    """
    Method for calculating the median pixel value for a given dataset.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, latitude, longitude
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: latitude, longitude
        variables: same as dataset_in
    """
    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    dataset_in_dtypes = None
    if dtype is None:
        # Save dtypes because masking with Dataset.where() converts to float64.
        band_list = list(dataset_in.data_vars)
        dataset_in_dtypes = {}
        for band in band_list:
            dataset_in_dtypes[band] = dataset_in[band].dtype

    # Mask out clouds and Landsat 7 scan lines.
    dataset_in = dataset_in.where((dataset_in != no_data) & (clean_mask))
    dataset_out = dataset_in.median(dim='time', skipna=True, keep_attrs=False)

    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out


def create_max_ndvi_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, intermediate_product=None, **kwargs):
    """
    Method for calculating the pixel value for the max ndvi value.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, latitude, longitude
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: latitude, longitude
        variables: same as dataset_in
    """
    dataset_in = dataset_in.copy(deep=True)

    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    dataset_in_dtypes = None
    if dtype is None:
        # Save dtypes because masking with Dataset.where() converts to float64.
        band_list = list(dataset_in.data_vars)
        dataset_in_dtypes = {}
        for band in band_list:
            dataset_in_dtypes[band] = dataset_in[band].dtype

    # Mask out clouds and scan lines.
    dataset_in = dataset_in.where((dataset_in != -9999) & clean_mask)

    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None

    time_slices = range(len(dataset_in.time))
    for timeslice in time_slices:
        dataset_slice = dataset_in.isel(time=timeslice).drop('time')
        ndvi = (dataset_slice.nir - dataset_slice.red) / (dataset_slice.nir + dataset_slice.red)
        ndvi.values[np.invert(clean_mask)[timeslice, ::]] = -1000000000
        dataset_slice['ndvi'] = ndvi
        if dataset_out is None:
            dataset_out = dataset_slice.copy(deep=True)
            utilities.clear_attrs(dataset_out)
        else:
            for key in list(dataset_slice.data_vars):
                dataset_out[key].values[dataset_slice.ndvi.values > dataset_out.ndvi.values] = \
                    dataset_slice[key].values[dataset_slice.ndvi.values > dataset_out.ndvi.values]
    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out


def create_min_ndvi_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, intermediate_product=None, **kwargs):
    """
    Method for calculating the pixel value for the min ndvi value.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, latitude, longitude
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: latitude, longitude
        variables: same as dataset_in
    """
    dataset_in = dataset_in.copy(deep=True)

    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    dataset_in_dtypes = None
    if dtype is None:
        # Save dtypes because masking with Dataset.where() converts to float64.
        band_list = list(dataset_in.data_vars)
        dataset_in_dtypes = {}
        for band in band_list:
            dataset_in_dtypes[band] = dataset_in[band].dtype

    # Mask out clouds and scan lines.
    dataset_in = dataset_in.where((dataset_in != -9999) & clean_mask)

    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None

    time_slices = range(len(dataset_in.time))
    for timeslice in time_slices:
        dataset_slice = dataset_in.isel(time=timeslice).drop('time')
        ndvi = (dataset_slice.nir - dataset_slice.red) / (dataset_slice.nir + dataset_slice.red)
        ndvi.values[np.invert(clean_mask)[timeslice, ::]] = 1000000000
        dataset_slice['ndvi'] = ndvi
        if dataset_out is None:
            dataset_out = dataset_slice.copy(deep=True)
            utilities.clear_attrs(dataset_out)
        else:
            for key in list(dataset_slice.data_vars):
                dataset_out[key].values[dataset_slice.ndvi.values <
                                        dataset_out.ndvi.values] = dataset_slice[key].values[dataset_slice.ndvi.values <
                                                                                             dataset_out.ndvi.values]
    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, None, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out

def unpack_bits(land_cover_endcoding, data_array, cover_type):
    """
	Description:
		Unpack bits for end of ls7 and ls8 functions 
	-----
	Input:
		land_cover_encoding(dict hash table) land cover endcoding provided by ls7 or ls8
        data_array( xarray DataArray)
        cover_type(String) type of cover
	Output:
        unpacked DataArray
	"""
    boolean_mask = np.isin(data_array.values, land_cover_endcoding[cover_type]) 
    return xr.DataArray(boolean_mask.astype(bool),
                        coords = data_array.coords,
                        dims = data_array.dims,
                        name = cover_type + "_mask",
                        attrs = data_array.attrs)

def ls8_unpack_qa( data_array , cover_type):

    land_cover_endcoding = dict( fill         =[1] ,
                                 clear        =[322, 386, 834, 898, 1346],
                                 water        =[324, 388, 836, 900, 1348],
                                 shadow       =[328, 392, 840, 904, 1350],
                                 snow         =[336, 368, 400, 432, 848, 880, 812, 944, 1352],
                                 cloud        =[352, 368, 416, 432, 848, 880, 912, 944, 1352],
                                 low_conf_cl  =[322, 324, 328, 336, 352, 368, 834, 836, 840, 848, 864, 880],
                                 med_conf_cl  =[386, 388, 392, 400, 416, 432, 898, 900, 904, 928, 944],
                                 high_conf_cl =[480, 992],
                                 low_conf_cir =[322, 324, 328, 336, 352, 368, 386, 388, 392, 400, 416, 432, 480],
                                 high_conf_cir=[834, 836, 840, 848, 864, 880, 898, 900, 904, 912, 928, 944],
                                 terrain_occ  =[1346,1348, 1350, 1352]
                               )
    return unpack_bits(land_cover_endcoding, data_array, cover_type)


def ls8_oli_unpack_qa(data_array, cover_type):
    """
    Returns a boolean `xarray.DataArray` denoting which points in `data_array`
    are of the selected `cover_type` (True indicates presence and
    False indicates absence).

    For more information, see this: https://landsat.usgs.gov/collectionqualityband
    The most relevant section for this function is titled
    "Landsat 8 OLI/ OLI-TIRS Level-1 Possible Attributes,
     Pixel Values, and Pixel Value Interpretations".

    Parameters
    ----------
    data_array: xarray.DataArray
        A DataArray of the QA band.
    cover_type: string
        A string in the set [fill, terrain_occ, clear, rad_sat_1_2,
                             rad_sat_3_4, rad_sat_5_pls, cloud, low_conf_cl,
                             med_conf_cl, high_conf_cl, high_cl_shdw,
                             high_snow_ice, low_conf_cir, high_conf_cir].

        'fill' removes "no_data" values, which indicates an absence of data. This value is -9999 for Landsat platforms.
        Generally, don't use 'fill'.
        'terrain_occ' allows only occluded terrain.
        'clear' allows only clear terrain. 'water' allows only water. 'shadow' allows only cloud shadows.
        'rad_sat_1_2'   denotes radiometric saturation in 1 or 2 bands.
        'rad_sat_3_4'   denotes radiometric saturation in 3 or 4 bands.
        'rad_sat_5_pls' denotes radiometric saturation in 5 or more bands.
        'cloud' allows only clouds, but note that it often only selects cloud boundaries.
        'low_conf_cl', 'med_conf_cl', and 'high_conf_cl' denote low, medium, and high confidence in cloud coverage.
        - 'low_conf_cl' is useful on its own for only removing clouds, however, 'clear' is usually better suited for this.
        - 'med_conf_cl' is useful in combination with 'low_conf_cl' to allow slightly heavier cloud coverage.
        - Note that 'med_conf_cl' and 'cloud' are very similar.
        - 'high_conf_cl' is useful in combination with both 'low_conf_cl' and 'med_conf_cl'.
        'high_cl_shdw' denotes high confidence in cloud shadow.
        'high_snow_ice' denotes high confidence in snow or ice.
        'low_conf_cir' and 'high_conf_cir' denote low and high confidence in cirrus clouds.

    Returns
    -------
    mask: xarray.DataArray
        The boolean `xarray.DataArray` denoting which points in `data_array`
        are of the selected `cover_type` (True indicates presence and
        False indicates absence). This will have the same dimensions and coordinates as `data_array`.
    """
    land_cover_encoding = dict(fill         =[1],
                               terrain_occ  =[2, 2722],
                               clear        =[2720, 2724, 2728, 2732],
                               rad_sat_1_2  =[2724, 2756, 2804, 2980, 3012, 3748, 3780, 6820, 6852, 6900, 7076, 7108, 7844, 7876],
                               rad_sat_3_4  =[2728, 2760, 2808, 2984, 3016, 3752, 3784, 6824, 6856, 6904, 7080, 7112, 7848, 7880],
                               rad_sat_5_pls=[2732, 2764, 2812, 2988, 3020, 3756, 3788, 6828, 6860, 6908, 7084, 7116, 7852, 7884],
                               cloud        =[2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908],
                               low_conf_cl  =[2752, 2722, 2724, 2728, 2732, 2976, 2980, 2984, 2988, 3744, 3748, 3752, 3756, 6816, 6820, 6824, 6828, 7072, 7076, 7080, 7084, 7840, 7844, 7848, 7852],
                               med_conf_cl  =[2752, 2756, 2760, 2764, 3008, 3012, 3016, 3020, 3776, 3780, 3784, 3788, 6848, 6852, 6856, 6860, 7104, 7108, 7112, 7116, 7872, 7876, 7880, 7884],
                               high_conf_cl =[2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908],
                               high_cl_shdw=[2976, 2980, 2984, 2988, 3008, 3012, 3016, 3020, 7072, 7076, 7080, 7084, 7104, 7108, 7112, 7116],
                               high_snow_ice=[3744, 3748, 3752, 3756, 3776, 3780, 3784, 3788, 7840, 7844, 7848, 7852, 7872, 7876, 7880, 7884],
                               low_conf_cir =[2720, 2722, 2724, 2728, 2732, 2752, 2756, 2760, 2764, 2800, 2804, 2808, 2812, 2976, 2980, 2984, 2988, 3008, 3012, 3016, 3020, 3744, 3748, 3752, 3756, 3780, 3784, 3788],
                               high_conf_cir=[6816, 6820, 6824, 6828, 6848, 6852, 6856, 6860, 6896, 6900, 6904, 6908, 7072, 7076, 7080, 7084, 7104, 7108, 7112, 7116, 7840, 7844, 7848, 7852, 7872, 7876, 7880, 7884]
                               )
    return unpack_bits(land_cover_encoding, data_array, cover_type)

def ls7_unpack_qa( data_array , cover_type):

    land_cover_endcoding = dict( fill     =  [1],
                                 clear    =  [66,  130],
                                 water    =  [68,  132],
                                 shadow   =  [72,  136],
                                 snow     =  [80,  112, 144, 176],
                                 cloud    =  [96,  112, 160, 176, 224],
                                 low_conf =  [66,  68,  72,  80,  96,  112],
                                 med_conf =  [130, 132, 136, 144, 160, 176],
                                 high_conf=  [224]
                               )
    return unpack_bits(land_cover_endcoding, data_array, cover_type)

def ls5_unpack_qa( data_array , cover_type):

    land_cover_endcoding = dict( fill     =  [1],
                                 clear    =  [66,  130],
                                 water    =  [68,  132],
                                 shadow   =  [72,  136],
                                 snow     =  [80,  112, 144, 176],
                                 cloud    =  [96,  112, 160, 176, 224],
                                 low_conf =  [66,  68,  72,  80,  96,  112],
                                 med_conf =  [130, 132, 136, 144, 160, 176],
                                 high_conf=  [224]
                               )
    return unpack_bits(land_cover_endcoding, data_array, cover_type)


def create_hdmedians_multiple_band_mosaic(dataset_in,
                                          clean_mask=None,
                                          no_data=-9999,
                                          dtype=None,
                                          intermediate_product=None,
                                          operation="median",
                                          **kwargs):
    """
    Calculates the geomedian or geomedoid using a multi-band processing method.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, latitude, longitude (in that order)
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.
    operation: str in ['median', 'medoid']

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: latitude, longitude
        variables: same as dataset_in
    """
    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)
    assert operation in ['median', 'medoid'], "Only median and medoid operations are supported."

    band_list = list(dataset_in.data_vars)
    dataset_in_dtypes = None
    if dtype is None:
        # Save dtypes because masking with Dataset.where() converts to float64.
        dataset_in_dtypes = {}
        for band in band_list:
            dataset_in_dtypes[band] = dataset_in[band].dtype

    # Mask out clouds and scan lines.
    dataset_in = dataset_in.where((dataset_in != no_data) & clean_mask)

    arrays = [dataset_in[band] for band in band_list]
    stacked_data = np.stack(arrays)
    bands_shape, time_slices_shape, lat_shape, lon_shape = stacked_data.shape[0], \
                                                           stacked_data.shape[1], stacked_data.shape[2], \
                                                           stacked_data.shape[3]
    # Reshape to remove lat/lon
    reshaped_stack = stacked_data.reshape(bands_shape, time_slices_shape,
                                          lat_shape * lon_shape)
    # Build zeroes array across time slices.
    hdmedians_result = np.zeros((bands_shape, lat_shape * lon_shape))

    # For each pixel (lat/lon combination), find the geomedian or geomedoid across time.
    for x in range(reshaped_stack.shape[2]):
        try:
            hdmedians_result[:, x] = hd.nangeomedian(
                reshaped_stack[:, :, x], axis=1) if operation == "median" else hd.nanmedoid(
                reshaped_stack[:, :, x], axis=1)
        except ValueError as e:
            # If all bands have nan values across time, the geomedians are nans.
            hdmedians_result[:, x] = np.full((bands_shape), np.nan)
    output_dict = {
        value: (('latitude', 'longitude'), hdmedians_result[index, :].reshape(lat_shape, lon_shape))
        for index, value in enumerate(band_list)
    }
    dataset_out = xr.Dataset(output_dict,
                             coords={'latitude': dataset_in['latitude'],
                                     'longitude': dataset_in['longitude']},
                             attrs=dataset_in.attrs)
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out

def restore_or_convert_dtypes(dtype_for_all=None, band_list=None, dataset_in_dtypes=None, dataset_out=None, no_data=-9999):
    """
    Converts datatypes of data variables in a copy of an xarray Dataset.

    Parameters
    ----------
    dtype_for_all: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.
    band_list: list-like
        !! UNUSED, DEPRECATED !!
    dataset_in_dtypes: dict
        A dictionary mapping band names to datatypes.
        One of `dtype_for_all` or `dataset_in_dtypes` must be `None`.
    no_data: int or float
        The no data value.

    Returns
    -------
    dataset_out: xarray.Dataset
        The output Dataset.
    """
    assert dtype_for_all is None or dataset_in_dtypes is None, \
        "One of `dtype_for_all` or `dataset_in_dtypes` must be `None`."
    if dtype_for_all is not None:
        # Integer types can't represent nan.
        if np.issubdtype(dtype_for_all, np.integer): # This also works for Python int type.
            utilities.nan_to_num(dataset_out, no_data)
        convert_to_dtype(dataset_out, dtype_for_all)
    else:  # Restore dtypes to state before masking.
        for band in dataset_in_dtypes:
            band_dtype = dataset_in_dtypes[band]
            if np.issubdtype(band_dtype, np.integer):
                utilities.nan_to_num(dataset_out[band], no_data)
            dataset_out[band] = dataset_out[band].astype(band_dtype)
    return dataset_out
