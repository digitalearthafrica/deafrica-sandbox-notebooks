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

from .dc_water_classifier import wofs_classify
from .dc_utilities import create_cfmask_clean_mask, create_bit_mask
from datetime import datetime

import scipy.ndimage.filters as conv
import numpy as np


def compute_coastal_change(old_mosaic, new_mosaic):
    """Compute the coastal change and coastlines for two mosaics

    Computes the output products and appends them onto the old mosaic as
    coastal_change, coastline_old, coastline_new

    Args:
        old_mosaic, new_mosaic - single timeslice mosaic data.

    Returns:
        Xarray dataset containing all original data with three new variables.

    """
    # Create a combined bitmask - cfmask if it exists, otherwise pixel_qa.
    combined_mask = create_cfmask_clean_mask(old_mosaic.cf_mask) & create_cfmask_clean_mask(
        new_mosaic.cf_mask) if 'cf_mask' in old_mosaic else create_bit_mask(
            old_mosaic.pixel_qa, [1, 2]) & create_bit_mask(new_mosaic.pixel_qa, [1, 2])

    old_water = wofs_classify(old_mosaic, mosaic=True, clean_mask=combined_mask)
    new_water = wofs_classify(new_mosaic, mosaic=True, clean_mask=combined_mask)

    coastal_change = new_water - old_water

    coastal_change = coastal_change.where(coastal_change.wofs != 0)

    new_coastline = _coastline_classification_2(new_water)
    old_coastline = _coastline_classification_2(old_water)

    old_mosaic['coastal_change'] = coastal_change.wofs
    old_mosaic['coastline_old'] = old_coastline.coastline
    old_mosaic['coastline_new'] = new_coastline.coastline

    return old_mosaic


def mask_mosaic_with_coastlines(dataset):
    """Mask a mosaic using old/new coastline"""

    required_measurements = ['red', 'green', 'blue', 'coastline_old', 'coastline_new']
    assert set(required_measurements).issubset(
        set(dataset.data_vars)), "Please include all required bands: Red, green, blue, and coastline masks."

    green = _darken_color([89, 255, 61], .8)
    pink = [[255, 8, 74], [252, 8, 74], [230, 98, 137], [255, 147, 172], [255, 192, 205]][0]
    blue = [[13, 222, 255], [139, 237, 236], [0, 20, 225], [30, 144, 255]][-1]
    dataset_clone = dataset.copy(deep=True)
    # mask the new coastline in blue.
    dataset_clone.red.values[dataset_clone.coastline_new.values == 1] = _adjust_color(blue[0])
    dataset_clone.green.values[dataset_clone.coastline_new.values == 1] = _adjust_color(blue[1])
    dataset_clone.blue.values[dataset_clone.coastline_new.values == 1] = _adjust_color(blue[2])
    #mask the old coastline in green.
    dataset_clone.red.values[dataset_clone.coastline_old.values == 1] = _adjust_color(green[0])
    dataset_clone.green.values[dataset_clone.coastline_old.values == 1] = _adjust_color(green[1])
    dataset_clone.blue.values[dataset_clone.coastline_old.values == 1] = _adjust_color(green[2])

    return dataset_clone


def mask_mosaic_with_coastal_change(dataset):
    """Mask a mosaic with coastal change"""

    required_measurements = ['red', 'green', 'blue', 'coastal_change']
    assert set(required_measurements).issubset(
        set(dataset.data_vars)), "Please include all required bands: Red, green, blue, and coastal change."

    green = _darken_color([89, 255, 61], .8)
    pink = [[255, 8, 74], [252, 8, 74], [230, 98, 137], [255, 147, 172], [255, 192, 205]][0]
    blue = [[13, 222, 255], [139, 237, 236], [0, 20, 225], [30, 144, 255]][-1]
    dataset_clone = dataset.copy(deep=True)
    dataset_clone.red.values[dataset_clone.coastal_change.values == 1] = _adjust_color(pink[0])
    dataset_clone.green.values[dataset_clone.coastal_change.values == 1] = _adjust_color(pink[1])
    dataset_clone.blue.values[dataset_clone.coastal_change.values == 1] = _adjust_color(pink[2])

    dataset_clone.red.values[dataset_clone.coastal_change.values == -1] = _adjust_color(green[0])
    dataset_clone.green.values[dataset_clone.coastal_change.values == -1] = _adjust_color(green[1])
    dataset_clone.blue.values[dataset_clone.coastal_change.values == -1] = _adjust_color(green[2])

    return dataset_clone


def _adjust_color(color, scale=4096):
    return int(float(color * scale) / 255.0)


def _darken_color(color, scale=0.8):
    return [int(float(x * scale)) for x in color]


def _coastline_classification(dataset, water_band='wofs'):
    kern = np.array([[1, 1, 1], [1, 0.001, 1], [1, 1, 1]])
    convolved = conv.convolve(dataset[water_band], kern, mode='constant') // 1

    ds = dataset.where(convolved > 0)
    ds = ds.where(convolved < 6)
    ds.wofs.values[~np.isnan(ds.wofs.values)] = 1
    ds.wofs.values[np.isnan(ds.wofs.values)] = 0
    ds.rename({"wofs": "coastline"}, inplace=True)

    return ds


def _coastline_classification_2(dataset, water_band='wofs'):
    kern = np.array([[1, 1, 1], [1, 0.001, 1], [1, 1, 1]])
    convolved = conv.convolve(dataset[water_band], kern, mode='constant', cval=-999) // 1

    ds = dataset.copy(deep=True)
    ds.wofs.values[(~np.isnan(ds[water_band].values)) & (ds.wofs.values == 1)] = 1
    ds.wofs.values[convolved < 0] = 0
    ds.wofs.values[convolved > 6] = 0
    ds.rename({"wofs": "coastline"}, inplace=True)

    return ds
