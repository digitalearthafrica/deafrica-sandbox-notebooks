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

import gdal, osr
import collections
import gc
import numpy as np
import xarray as xr
from datetime import datetime
import collections
from collections import OrderedDict

import datacube
from . import dc_utilities as utilities
from .dc_utilities import create_default_clean_mask

def geometric_median(x, epsilon=1, max_iter=40):
    """
    Calculates the geometric median of band reflectances
    The procedure stops when either the error tolerance 'tol' or the maximum number of iterations 'MaxIter' is reached. 
    Args:
        x: (p x N) matrix, where p = number of bands and N = number of dates during the period of interest
        max_iter: maximum number of iterations
        tol: tolerance criterion to stop iteration   
    
    Returns:
        geo_median: p-dimensional vector with geometric median reflectances
    """
    y0 = np.nanmean(x, axis=1)
    if len(y0[np.isnan(y0)]) > 0:
        return y0

    for _ in range(max_iter):
        euc_dist = np.transpose(np.transpose(x) - y0)
        euc_norm = np.sqrt(np.sum(euc_dist ** 2, axis=0))
        not_nan = np.where(~np.isnan(euc_norm))[0]
        y1 = np.sum(x[:, not_nan] / euc_norm[not_nan], axis=1) / (np.sum(1 / euc_norm[not_nan]))
        if len(y1[np.isnan(y1)]) > 0 or np.sqrt(np.sum((y1 - y0) ** 2)) < epsilon:
            return y1

        y0 = y1

    return y0

try:
    import hdmedians as hd
except: 
    
    hd = geometric_median


def create_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
    Description:
      Creates a most recent - oldest mosaic of the input dataset. If no clean mask is given,
      the 'cf_mask' variable must be included in the input dataset, as it will be used
      to create a clean mask
    -----
    Inputs:
      dataset_in (xarray.Dataset) - dataset retrieved from the Data Cube; should contain
        coordinates: time, latitude, longitude
        variables: variables to be mosaicked
        If user does not provide a clean_mask, dataset_in must also include the cf_mask
        variable
    Optional Inputs:
      clean_mask (nd numpy array with dtype boolean) - true for values user considers clean;
        if user does not provide a clean mask, one will be created using cfmask
      no_data (int/float) - no data pixel value; default: -9999
    Output:
      dataset_out (xarray.Dataset) - mosaicked data with
        coordinates: latitude, longitude
        variables: same as dataset_in
    """

    dataset_in = dataset_in.copy(deep=True)

    assert clean_mask is not None, "Please provide a boolean clean mask."

    #masks data with clean_mask. all values that are clean_mask==False are set to nodata.
    for key in list(dataset_in.data_vars):
        dataset_in[key].values[np.invert(clean_mask)] = no_data
    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None
    time_slices = reversed(range(len(dataset_in.time))) if kwargs and kwargs['reverse_time'] else range(
        len(dataset_in.time))
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
    return dataset_out


def create_mean_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
	Description:
		Method for calculating the mean pixel value for a given dataset.
	-----
	Input:
		dataset_in (xarray dataset) - the set of data with clouds and no data removed.
	Optional Inputs:
		no_data (int/float) - no data value.
	"""
    assert clean_mask is not None, "A boolean mask for clean_mask must be supplied."

    dataset_in_filtered = dataset_in.where((dataset_in != no_data) & (clean_mask))
    dataset_out = dataset_in_filtered.mean(dim='time', skipna=True, keep_attrs=False)
    utilities.nan_to_num(dataset_out, no_data)
    #manually clear out dates/timestamps/sats.. median won't produce meaningful reslts for these.
    for key in ['timestamp', 'date', 'satellite']:
        if key in dataset_out:
            dataset_out[key].values[::] = no_data
    return dataset_out.astype(kwargs.get('dtype', 'int32'))


def create_median_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
	Description:
		Method for calculating the median pixel value for a given dataset.
	-----
	Input:
		dataset_in (xarray dataset) - the set of data with clouds and no data removed.
	Optional Inputs:
		no_data (int/float) - no data value.
	"""
    assert clean_mask is not None, "A boolean mask for clean_mask must be supplied."

    dataset_in_filtered = dataset_in.where((dataset_in != no_data) & (clean_mask))
    dataset_out = dataset_in_filtered.median(dim='time', skipna=True, keep_attrs=False)
    utilities.nan_to_num(dataset_out, no_data)
    #manually clear out dates/timestamps/sats.. median won't produce meaningful reslts for these.
    for key in ['timestamp', 'date', 'satellite']:
        if key in dataset_out:
            dataset_out[key].values[::] = no_data
    return dataset_out.astype(kwargs.get('dtype', 'int32'))


def create_max_ndvi_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
	Description:
		Method for calculating the pixel value for the max ndvi value.
	-----
	Input:
		dataset_in (xarray dataset) - the set of data with clouds and no data removed.
	Optional Inputs:
		no_data (int/float) - no data value.
	"""

    dataset_in = dataset_in.copy(deep=True)

    assert clean_mask is not None, "Please provide a boolean clean mask."

    for key in list(dataset_in.data_vars):
        clean_mask_temp = clean_mask.values if isinstance(clean_mask, xr.DataArray)\
                          else clean_mask
        dataset_in[key].values[np.invert(clean_mask.values)] = no_data

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
                dataset_out[key].values[dataset_slice.ndvi.values >
                                        dataset_out.ndvi.values] = dataset_slice[key].values[dataset_slice.ndvi.values >
                                                                                             dataset_out.ndvi.values]
    return dataset_out

def create_min_ndvi_mosaic(dataset_in, clean_mask=None, no_data=-9999, intermediate_product=None, **kwargs):
    """
	Description:
		Method for calculating the pixel value for the min ndvi value.
	-----
	Input:
		dataset_in (xarray dataset) - the set of data with clouds and no data removed.
	Optional Inputs:
		no_data (int/float) - no data value.
	"""

    dataset_in = dataset_in.copy(deep=True)

    assert clean_mask is not None, "Please provide a boolean clean mask."

    for key in list(dataset_in.data_vars):
        dataset_in[key].values[np.invert(clean_mask)] = no_data

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
    return xr.DataArray(boolean_mask.astype(np.int8),
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
    
def nan_to_num(dataset, number):
    for key in list(dataset.data_vars):
        dataset[key].values[np.isnan(dataset[key].values)] = number  
        
def create_hdmedians_multiple_band_mosaic(dataset_in,
                                          clean_mask=None,
                                          no_data=-9999,
                                          intermediate_product=None,
                                          operation="median",
                                          **kwargs):
        
    assert clean_mask is not None, "A boolean mask for clean_mask must be supplied."
    assert operation in ['median', 'medoid'], "Only median and medoid operations are supported."

    dataset_in_filtered = dataset_in.where((dataset_in != no_data) & (clean_mask))

    band_list = list(dataset_in_filtered.data_vars)
    arrays = [dataset_in_filtered[band] for band in band_list]

    stacked_data = np.stack(arrays)
    bands_shape, time_slices_shape, lat_shape, lon_shape = stacked_data.shape

    reshaped_stack = stacked_data.reshape(bands_shape, time_slices_shape,
                                          lat_shape * lon_shape)  # Reshape to remove lat/lon
    hdmedians_result = np.zeros((bands_shape, lat_shape * lon_shape))  # Build zeroes array across time slices.

    for x in range(reshaped_stack.shape[2]):
        try:
            hdmedians_result[:, x] = hd.nangeomedian(
                reshaped_stack[:, :, x], axis=1) if operation == "median" else hd.nanmedoid(
                    reshaped_stack[:, :, x], axis=1)
        except ValueError:
            no_data_pixel_stack = reshaped_stack[:, :, x]
            no_data_pixel_stack[np.isnan(no_data_pixel_stack)] = no_data
            hdmedians_result[:, x] = np.full((bands_shape), no_data) if operation == "median" else hd.nanmedoid(
                no_data_pixel_stack, axis=1)

    output_dict = {
        value: (('latitude', 'longitude'), hdmedians_result[index, :].reshape(lat_shape, lon_shape))
        for index, value in enumerate(band_list)
    }

    dataset_out = xr.Dataset(output_dict,
                             coords={'latitude': dataset_in['latitude'], 'longitude': dataset_in['longitude']},
                             attrs = dataset_in.attrs)
    nan_to_num(dataset_out, no_data)
    #return dataset_out
    return dataset_out.astype(kwargs.get('dtype', 'int32'))
