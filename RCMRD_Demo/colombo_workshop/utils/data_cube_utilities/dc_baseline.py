import numpy as np
import xarray
import xarray.core.ops as ops
import xarray as xr
from itertools import islice


def _window(seq, n=2):
    """Returns a sliding _window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def _composite_of_first(arrays, reverse=False, name_suffix="_composite"):
    #In memory of Rube Goldberg
    narrays = arrays.copy(deep=True)
    narrays.values = narrays.values[::-1] if reverse else narrays.values
    notnulls = [ops.notnull(array) for array in narrays]
    first_notnull = ops.argmax(ops.stack(notnulls), axis=0)
    composite = np.choose(first_notnull, narrays)
    return xr.DataArray(
        composite,
        coords=[narrays.latitude, narrays.longitude],
        dims=['latitude', 'longitude'],
        name="{band}{suffix}".format(band=narrays.name, suffix=name_suffix))


def _mosaic(dataset, most_recent_first=False, custom_label="_composite"):
    return xr.merge([
        _composite_of_first(dataset[variable], reverse=most_recent_first, name_suffix=custom_label)
        for variable in dataset.data_vars
    ])


def _composite_by_average(dataset, custom_label="_composite"):
    composite = dataset.mean('time')
    return composite


## This should be the the only method called from dc baseline
def generate_baseline(dataset, composite_size=5, mode="average", custom_label=""):
    ranges = _window(range(len(dataset.time)), n=composite_size + 1)
    reffs = (dataset.isel(time=frame[:-1]) for frame in ranges)

    baselines = None
    if mode == "average":
        baselines = (_composite_by_average(ref, custom_label=custom_label) for ref in reffs)
    elif mode == "composite":
        baselines = (_mosaic(ref, most_recent_first=True, custom_label=custom_label) for ref in reffs)

    baseline = xr.concat(baselines, dim='time')
    baseline['time'] = dataset.time[composite_size:]
    return baseline
