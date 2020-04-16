import numpy as np
import xarray as xr
from clean_mask import create_circular_mask
from skimage.filters.rank import modal
from skimage.morphology import remove_small_objects
import scipy

## Selective Filters (do not necessarily apply to all pixels) ##

def lone_object_filter(image, min_size=2, connectivity=1, kernel_size=3):
    """
    Replaces isolated, contiguous regions of values in a raster with values
    representing the surrounding pixels.

    More specifically, this reduces noise in a raster by setting
    contiguous regions of values of a specified minimum size to
    the modal value in a specified neighborhood.

    The default argument values filter out lone, singular pixels.
    This filter is not idempotent, so it may need to be applied repeatedly
    until the output stops changing or the results are acceptable.

    Args:
        image (numpy.ndarray):
            The image to filter.
        min_size (int):
            Defines the minimum number of contiguous pixels that will not
            be set to the modal value of their neighborhood. Setting this to 1
            is pointless, since this function will then do nothing to the raster.
        connectivity (int):
            The maximum distance between any two pixels such that they are
            considered one group. For example, a connectivity of 1 considers
            only adjacent values to be within one group, but a connectivity of 2
            also considers diagonally connected values to be within one group.
        kernel_size (int or float):
            The diameter of the circular kernel to use for the modal filter.
            If there are still large, contiguous regions of lone pixels,
            increase this value to remove them. Note that the larger this value
            is, the more detail may be lost in the image.

    Returns:
        The filtered image.

    Authors:
        Andrew Lubawy (andrew.m.lubawy@ama-inc.com)\n
        John Rattz    (john.c.rattz@ama-inc.com)
    """
    assert kernel_size % 2 == 1, "The parameter `kernel_size` must be an odd number."
    modal_filtered = modal(image, create_circular_mask(kernel_size, kernel_size))

    da = xr.DataArray(image)
    for i, val in enumerate(np.unique(image)):
        layer = remove_small_objects(image == val, min_size=min_size, connectivity=connectivity)
        filtered = da.where(layer) if i == 0 else filtered.combine_first(da.where(layer))
    filtered.values[np.isnan(filtered.values)] = modal_filtered[np.isnan(filtered.values)]
    return filtered.values

## End Selective Filters ##

## Non-Selective Filters (apply to all pixels) ##

def apply_filter(statistic, filter_output, padded_arr, filter_shape):
    """
    Creates a mean, median, or standard deviation filtered version
    of an `xarray.DataArray`.

    Parameters
    ----------
    filter_output: xarray.DataArray
        The `xarray.DataArray` to store the filtered values in.
        Must contain the values to filter. This object is modified.**
    statistic: string
        The name of the statistic to use for the filter.
        The possible values are ['mean', 'median', 'std'].
    padded_arr: numpy.ndarray
        A NumPy array with a shape matching `filter_output` padded for
        `filter_shape`.
    filter_shape: list-like of int
        A list-like of 2 positive integers defining the shape of the filter kernel.
    """
    # For each point in the first two dimensions of `dataarray`...
    for i in range(filter_output.shape[0]):
        for j in range(filter_output.shape[1]):
            padded_arr_segment = padded_arr[i:i + filter_shape[0],
                                 j:j + filter_shape[1]]
            if statistic == 'mean':
                filter_output.values[i, j] = np.nanmean(padded_arr_segment)
            elif statistic == 'median':
                filter_output.values[i, j] = np.nanmedian(padded_arr_segment)
            elif statistic == 'std':
                filter_output.values[i, j] = np.nanstd(padded_arr_segment)
    return filter_output


def stats_filter_3d_composite_2d(dataarray, statistic, filter_size=1,
                                 time_dim='time'):
    """
    Returns a mean, median, or standard deviation filter composite
    of a 3D `xarray.DataArray` with a time dimension. This makes a 2D composite
    of a 3D array by stretching the filter kernel across time.
    This function is more accurate than using SciPy or scikit-image methods, because
    those don't handle the extremities ideally (edges and corners).
    Specifically, only values actually inside the filter should be considered,
    so the data is padded with NaNs when "convolving" `dataarray` with an
    equally-weighted, rectangular kernel of shape `(filter_size, filter_size)`.
    This function is resilient to NaNs.

    Parameters
    ----------
    dataarray: xarray.DataArray
        The data to create a filtered version of. Must have 3 dimensions, with
        the last being 'time'.
    statistic: string
        The name of the statistic to use for the filter.
        The possible values are ['mean', 'median', 'std'].
    filter_size: int
        The size of the filter to use. Must be positive and should be odd.
        The filter shape will be `(filter_size, filter_size)`.
    time_dim: str
        The string name of the time dimension.
    """
    time_ax_num = dataarray.get_axis_num(time_dim)
    dataarray_non_time_dims = np.concatenate((np.arange(time_ax_num),
                                              np.arange(time_ax_num + 1, len(dataarray.dims))))
    filter_dims = np.array(dataarray.dims)[dataarray_non_time_dims]
    filter_coords = {dim: dataarray.coords[dim] for dim in filter_dims}
    filter_output = xr.DataArray(np.full(np.array(dataarray.shape)[dataarray_non_time_dims], np.nan),
                                 coords=filter_coords, dims=filter_dims)
    if filter_size == 1:
        agg_func_kwargs = dict(a=dataarray.values, axis=time_ax_num)
        if statistic == 'mean':
            filter_output.values[:] = np.nanmean(**agg_func_kwargs)
        elif statistic == 'median':
            filter_output.values[:] = np.nanmedian(**agg_func_kwargs)
        elif statistic == 'std':
            filter_output.values[:] = np.nanstd(**agg_func_kwargs)
    else:
        # Allocate a Numpy array containing the content of `dataarray`, but padded
        # with NaNs to ensure the statistics are correct at the x and y extremities of the data.
        flt_shp = np.array((filter_size, filter_size))
        shp = np.array(dataarray.shape)[dataarray_non_time_dims]
        pad_shp = (*(shp + flt_shp - 1), dataarray.shape[time_ax_num])
        padding = (flt_shp - 1) // 2  # The number of NaNs from an edge of the padding to the data.
        padded_arr = np.full(pad_shp, np.nan)
        padded_arr[padding[0]:pad_shp[0] - padding[0],
        padding[1]:pad_shp[1] - padding[1]] = dataarray.values
        filter_output = apply_filter(statistic, filter_output, padded_arr, flt_shp)
    return filter_output


def stats_filter_2d(dataarray, statistic, filter_size=3):
    """
    Returns a mean, median, or standard deviation filter of a 2D `xarray.DataArray`.
    This function is more accurate than using SciPy or scikit-image methods, because
    those don't handle the extremities ideally (edges and corners).
    Specifically, only values actually inside the filter should be considered,
    so the data is padded with NaNs when "convolving" `dataarray` with an
    equally-weighted, rectangular kernel of shape `(filter_size, filter_size)`.
    This function is resilient to NaNs.

    Parameters
    ----------
    dataarray: xarray.DataArray
        The data to create a filtered version of. Must have 2 dimensions.
    statistic: string
        The name of the statistic to use for the filter.
        The possible values are ['mean', 'median', 'std'].
    filter_size: int
        The size of the filter to use. Must be positive and should be odd.
        The filter shape will be `(filter_size, filter_size)`.
    """
    if filter_size == 1: return dataarray

    filter_output = dataarray.copy()
    kernel = np.ones((filter_size, filter_size))
    if statistic == 'mean':
        filter_output.values[:] = scipy.signal.convolve2d(filter_output.values, kernel, mode="same") / kernel.size
    elif statistic == 'median':
        filter_output.values[:] = scipy.ndimage.median_filter(filter_output.values, footprint=kernel)
    elif statistic == 'std':
        # Calculate standard deviation as stddev(X) = sqrt(E(X^2) - E(X)^2).
        im = dataarray.values
        im2 = im ** 2
        ones = np.ones(im.shape)

        s = scipy.signal.convolve2d(im, kernel, mode="same")
        s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
        ns = scipy.signal.convolve2d(ones, kernel, mode="same")

        filter_output.values[:] = np.sqrt((s2 - s ** 2 / ns) / ns)
    return filter_output

## End Non-Selective Filters ##
