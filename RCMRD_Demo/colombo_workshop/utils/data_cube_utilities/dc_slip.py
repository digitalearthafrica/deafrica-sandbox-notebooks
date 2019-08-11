import numpy as np
import xarray as xr


def compute_slip(baseline_data, target_data, dem_data):
    """Compute the SLIP product for a baseline and target dataset

    Args:
        baseline_data: single timeslice dataset with dims lat, lon
        target_data: single timeslice dataset with dims lat, lon
        dem_data: dem dataset for the above datasets with corresponding lat, lon indices

    Returns:
        xarray dataarray containing slip mask
    """

    required_measurements = ['red', 'nir', 'swir1']
    assert set(required_measurements).issubset(
        set(baseline_data.data_vars)), "Please include all required bands: Red, NIR, SWIR1."
    assert set(required_measurements).issubset(
        set(target_data.data_vars)), "Please include all required bands: Red, NIR, SWIR1."

    slip_mask = xr.Dataset(
        {
            'slip': (tuple(target_data.dims.keys()), np.ones(target_data.red.values.shape))
        },
        coords={'latitude': target_data.latitude,
                'longitude': target_data.longitude})

    filtered_baseline_data = baseline_data.where(baseline_data != -9999)

    ndwi_target = (target_data.nir - target_data.swir1) / (target_data.nir + target_data.swir1)
    ndwi_baseline = (filtered_baseline_data.nir - filtered_baseline_data.swir1) / (
        filtered_baseline_data.nir + filtered_baseline_data.swir1)
    ndwi_change = ndwi_target - ndwi_baseline
    target_ndwi_filtered = slip_mask.where(abs(ndwi_change) > 0.20)

    red_change = (target_data.red - filtered_baseline_data.red) / (filtered_baseline_data.red)
    target_red_filtered = target_ndwi_filtered.where(red_change > 0.40)

    is_above_slope_threshold = create_slope_mask(dem_data, degree_threshold=15, resolution=30)
    target_red_slope_filtered = target_red_filtered.where(is_above_slope_threshold)

    slip_mask.slip.values[target_red_slope_filtered.isnull().slip.values] = 0
    return slip_mask.slip.astype('int16')


def mask_mosaic_with_slip(dataset):
    """Using a mosaic and the slip mask, create a red masked dataset to be written to a png.
    """
    required_measurements = ['red', 'green', 'blue', 'slip']
    assert set(required_measurements).issubset(
        set(dataset.data_vars)), "Please include all required bands: Red, green, blue, and slip mask."

    masked_dataset = dataset.copy(deep=True)
    masked_dataset.red.values[masked_dataset.slip.values == 1] = 4096
    masked_dataset.green.values[masked_dataset.slip.values == 1] = 0
    masked_dataset.blue.values[masked_dataset.slip.values == 1] = 0

    return masked_dataset


def _generate_gradient(matrix, resolution=1.0, remove_border=False):
    north, east = np.gradient(matrix)
    max_gradient = np.maximum.reduce([abs(north), abs(east)])
    if remove_border:
        max_gradient[:, 0] = np.nan
        max_gradient[:, -1] = np.nan
        max_gradient[0, :] = np.nan
        max_gradient[-1, :] = np.nan
    return (max_gradient / float(resolution))


def _generate_degree(dem_matrix, resolutution=1.0):
    return np.rad2deg(np.arctan(_generate_gradient(dem_matrix, resolution=resolution)))


def create_slope_mask(dem_data, resolution=1.0, degree_threshold=15, no_data=-9999):
    ## Uses values at first DEM acquistion date
    target = dem_data.dem.values[0].astype(np.float32)
    target[target == no_data] = np.nan
    ## Generates gradient per dem pixel, turns to degrees per dem pixel, bounds to range between 1 and 100
    slopes = _generate_gradient(target, resolution=resolution)
    angle_of_elevation = np.rad2deg(np.arctan(slopes))
    ## Create a mask for greater than 15 degrees. Here is what 15 degrees looks like: https://i.stack.imgur.com/BIrAW.png
    mask = angle_of_elevation > degree_threshold
    return mask
