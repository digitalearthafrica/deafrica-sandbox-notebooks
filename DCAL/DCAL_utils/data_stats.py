import xarray as xr
from dc_time import _n64_datetime_to_scalar

## Data Availability ##

def find_gaps(data_arr, aggregation_method):
    """
    Finds the minimum, mean, median, or maximum time difference between True values
    in a boolean xarray.DataArray. This should be a faster implementation.

    Parameters
    ----------
    data_arr: xarray.DataArray of bool
        DataArray of boolean values denoting which elements are desired.
        Examples of desired elements include clear views (or "non-cloud pixels").
        This DataArray must have a 'time' dimension.
    aggregation_method: str
        The aggregation method to use. Can be any of ['min', 'mean', 'median', 'max'].

    Returns
    -------
    gaps: xarray.DataArray of float64
        The time gaps between True values in `data_arr`. Due to limitations of the numpy.datetime64 data type,
        the time differences are in seconds, stored as np.float64.
    """
    time_index = 0
    # 1. Convert time from numpy.datetime64 to scalars and broadcast along latitude and longitude.
    time = _n64_datetime_to_scalar(data_arr.time)
    time, _, _ = xr.broadcast(time, data_arr.latitude, data_arr.longitude)
    # 2. Fill each undesired point with its previous desired point's time and find the time differences.
    mask = data_arr == 1
    time = time.where(mask)
    time = time.ffill(dim='time')
    time_diff = time.diff('time')
    # A time difference is only 0 because of differencing after the forward fill.
    time_diff = time_diff.where(time_diff != 0)
    # 3. Calculate the desired statistic for the time differences.
    if aggregation_method == 'min':    return time_diff.min('time')
    if aggregation_method == 'mean':   return time_diff.mean('time')
    if aggregation_method == 'median': return time_diff.median('time')
    if aggregation_method == 'max':    return time_diff.max('time')

## End Data Availability ##