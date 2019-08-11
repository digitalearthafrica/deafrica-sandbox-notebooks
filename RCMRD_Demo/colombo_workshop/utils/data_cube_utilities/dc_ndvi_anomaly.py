from .dc_water_classifier import wofs_classify
import xarray as xr
import numpy as np


def compute_ndvi_anomaly(baseline_data,
                         scene_data,
                         baseline_clear_mask=None,
                         selected_scene_clear_mask=None,
                         nodata=-9999):
    """Compute the scene+baseline median ndvi values and the difference

    Args:
        basleine_data: xarray dataset with dims lat, lon, t
        scene_data: xarray dataset with dims lat, lon - should be mosaicked already.
        baseline_clear_mask: boolean mask signifying clear pixels for the baseline data
        selected_scene_clear_mask: boolean mask signifying lcear pixels for the baseline data
        nodata: nodata value for the datasets

    Returns:
        xarray dataset with scene_ndvi, baseline_ndvi(median), ndvi_difference, and ndvi_percentage_change.
    """

    assert selected_scene_clear_mask is not None and baseline_clear_mask is not None, "Both the selected scene and baseline data must have associated clear mask data."

    #cloud filter + nan out all nodata.
    baseline_data = baseline_data.where((baseline_data != nodata) & baseline_clear_mask)

    baseline_ndvi = (baseline_data.nir - baseline_data.red) / (baseline_data.nir + baseline_data.red)
    median_ndvi = baseline_ndvi.median('time')

    #scene should already be mosaicked.
    water_class = wofs_classify(scene_data, clean_mask=selected_scene_clear_mask, mosaic=True).wofs
    scene_cleaned = scene_data.copy(deep=True).where((scene_data != nodata) & (water_class == 0))
    scene_ndvi = (scene_cleaned.nir - scene_cleaned.red) / (scene_cleaned.nir + scene_cleaned.red)

    ndvi_difference = scene_ndvi - median_ndvi
    ndvi_percentage_change = (scene_ndvi - median_ndvi) / median_ndvi

    #convert to conventional nodata vals.
    scene_ndvi.values[~np.isfinite(scene_ndvi.values)] = nodata
    ndvi_difference.values[~np.isfinite(ndvi_difference.values)] = nodata
    ndvi_percentage_change.values[~np.isfinite(ndvi_percentage_change.values)] = nodata

    scene_ndvi_dataset = xr.Dataset(
        {
            'scene_ndvi': scene_ndvi,
            'baseline_ndvi': median_ndvi,
            'ndvi_difference': ndvi_difference,
            'ndvi_percentage_change': ndvi_percentage_change
        },
        coords={'latitude': scene_data.latitude,
                'longitude': scene_data.longitude})

    return scene_ndvi_dataset
