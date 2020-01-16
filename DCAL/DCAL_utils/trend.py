from functools import partial  
from itertools import islice, product  
import numpy as np
import xarray as xr

def __where_not_nan(arr: np.ndarray):
    """Finds position of not nan values in an nd-array

    Args:
        arr (numpy.ndarray): nd-array with nan values
        
    Returns:
        data (xr.DataArray): nd-array with indices of finite(not nan) values
    """
    return np.where(np.isfinite(arr))


def __flatten_shallow(arr):
    """Flattens first two axes of nd-array
    Args:
        arr (numpy.ndarray): nd-array with dimensions (n, m)
        
    Returns: 
        arr (numpy.ndarray): nd-array with dimensions (n*m)
    """
    # TODO: Done in a hurry, Find numpy native way of resizing 
    return arr.reshape(arr.shape[0] * arr.shape[1])


def __linear_fit(da: xr.DataArray):
    """Applies linear regression on a 1-D xr.DataArray. 

    Args:
        da (xr.DataArray): 1-D Data-Array being manipulated. 
        
    Returns:
        data (xr.DataArray): DataArray with a single element(slope of regression).
    """
    
    xs = np.array(list(range(len(da.time))))
    ys = __flatten_shallow(da.values)
        
    not_nan = __where_not_nan(ys)[0].astype(int)

    xs = xs[not_nan]
    ys = ys[not_nan] 

    pf = np.polyfit(xs,ys, 1)
    return xr.DataArray(pf[0])


def linear(da: xr.DataArray):
    """Reduces xarray along a time component. The reduction yields a slope for each spatial coordinate in the xarray. 

    Args:
        da (xr.DataArray): 3-D Data-Array being manipulated. `latitude` and `longitude` are required dimensions.

    Returns:
        linear_trend_product (xr.DataArray): 2-D Data-Array
    """

    # TODO: Decouple from coordinate system, and allow regression along multiple components.
    stacked = da.stack(allpoints = ['latitude',
                                    'longitude'])
    
    trend = stacked.groupby('allpoints').apply(__linear_fit)
    
    unstacked = trend.unstack('allpoints')
    
    return unstacked.rename(dict(allpoints_level_0 = "latitude",
                                 allpoints_level_1 = "longitude"))