#Make a function for this since we will be doing this multiple times
from utils.data_cube_utilities.dc_mosaic import (ls7_unpack_qa, ls8_unpack_qa, ls5_unpack_qa)
import numpy as np


def landsat_qa_clean_mask(dataset, platform):
    """
    Returns a clean_mask for `dataset` that masks out clouds.
    
    Parameters
    ----------
    dataset: xarray.Dataset
        An xarray (usually produced by `datacube.load()`) that contains a `pixel_qa` data variable.
    platform: str
        A string denoting the platform to be used. Can be "LANDSAT_5", "LANDSAT_7", or "LANDSAT_8".
    """
    processing_options = {
        "LANDSAT_5": ls5_unpack_qa,
        "LANDSAT_7": ls7_unpack_qa,
        "LANDSAT_8": ls8_unpack_qa
    }
    
    #Clean mask creation to filter out pixels that are not suitable for analysis
    clear_xarray  = processing_options[platform](dataset.pixel_qa, "clear")  
    water_xarray  = processing_options[platform](dataset.pixel_qa, "water")
    
    #use logical or statement to elect viable pixels for analysis
    return np.logical_or(clear_xarray.values.astype(bool), water_xarray.values.astype(bool))


# def landsat_cf_clean_mask(dataset, platform):
    
#     processing_options = {
#         "LANDSAT_8": l8_clean_mask,
#         "LANDSAT_7": ls7_unpack_qa
#     }
    
#     #Clean mask creation to filter out pixels that are not suitable for analysis
#     clear_xarray  = processing_options[platform](dataset.pixel_qa, "clear")  
#     water_xarray  = processing_options[platform](dataset.pixel_qa, "water")
    
#     #use logical or statement to elect viable pixels for analysis
#     return np.logical_or(clear_xarray.values.astype(bool), water_xarray.values.astype(bool))