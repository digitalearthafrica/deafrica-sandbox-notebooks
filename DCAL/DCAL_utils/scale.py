import xarray as xr
import numpy as np

def xr_scale(data, data_vars=None, min_max=None, scaling='norm', copy=False):
    """
    Scales an xarray Dataset or DataArray with standard scaling or norm scaling.
    
    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray
        The NumPy array to scale.
    data_vars: list
        The names of the data variables to scale.
    min_max: tuple
        A 2-tuple which specifies the desired range of the final output - the minimum and the maximum, in that order.
        If all values are the same, all values will become min_max[0].
    scaling: str
        The options are ['std', 'norm']. 
        The option 'std' standardizes. The option 'norm' normalizes (min-max scales). 
    copy: bool
        Whether or not to copy `data` before scaling.
    """
    data = data.copy() if copy else data
    if isinstance(data, xr.Dataset):
        data_arr_names = list(data.data_vars) if data_vars is None else data_vars
        for data_arr_name in data_arr_names:
            data_arr = data[data_arr_name]
            data_arr.values = np_scale(data_arr.values, min_max=min_max, scaling=scaling)
    elif isinstance(data, xr.DataArray): 
        data.values = np_scale(data.values, min_max=min_max, scaling=scaling)
    return data


def np_scale(arr, pop_arr=None, pop_min_max=None, pop_mean_std=None, min_max=None, scaling='norm'):
    """
    Scales a NumPy array with standard scaling or norm scaling, default to norm scaling.

    Parameters
    ----------
    arr: numpy.ndarray
        The NumPy array to scale.
    pop_arr: numpy.ndarray, optional
        The NumPy array to treat as the population.
        If specified, all members of `arr` must be within the range of `pop_arr`
        or `min_max` must be specified.
    pop_min_max: list-like, optional
        The population minimum and maximum, in that order.
        Supercedes `pop_arr` when normalizing.
    pop_mean_std: list-like, optional
        The population mean and standard deviation, in that order.
        Supercedes `pop_arr` when standard scaling.
    min_max: list-like, optional
        The desired minimum and maximum of the final output, in that order.
        If all values are the same, all values will become `min_max[0]`.
    scaling: str, optional
        The options are ['std', 'norm'].
        The option 'std' standardizes. The option 'norm' normalizes (min-max scales).
    """
    if len(arr) == 0:
        return arr
    pop_arr = arr if pop_arr is None else pop_arr
    if scaling == 'norm':
        pop_min, pop_max = (pop_min_max[0], pop_min_max[1]) if pop_min_max is not None \
            else (np.nanmin(pop_arr), np.nanmax(pop_arr))
        numerator, denominator = arr - pop_min, pop_max - pop_min
    elif scaling == 'std':
        mean, std = pop_mean_std if pop_mean_std is not None else (np.nanmean(pop_arr), np.nanstd(pop_arr))
        numerator, denominator = arr - mean, std
    # Primary scaling
    new_arr = arr
    if denominator > 0:
        new_arr = numerator / denominator
    # Optional final scaling.
    if min_max is not None:
        if denominator > 0:
            new_arr = np.interp(new_arr, (np.nanmin(new_arr), np.nanmax(new_arr)), min_max)
        else: # The values are identical - set all values to the low end of the desired range.
            new_arr = np.full_like(new_arr, min_max[0])
    return new_arr