import matplotlib.pyplot as plt
from time import time
import numpy as np

from plotter_utils import figure_ratio, xarray_set_axes_labels, retrieve_or_create_fig_ax

# Change the bands (RGB) here if you want other false color combinations
def rgb(dataset, at_index=0, x_coord='longitude', y_coord='latitude',
        bands=['red', 'green', 'blue'], paint_on_mask = [],
        min_possible=0, max_possible=10000, use_data_min=False,
        use_data_max=False, min_inten=0.15, max_inten=1.0,
        width=10, fig=None, ax=None, imshow_kwargs=None):
    """
    Creates a figure showing an area, using three specified bands as the rgb componenets.

    Parameters
    ----------
    dataset: xarray.Dataset
        A Dataset containing at least latitude and longitude coordinates and optionally time.
        The coordinate order should be time, latitude, and finally longitude.
        Must contain the data variables specified in the `bands` parameter.
    at_index: int
        The time index to show.
    x_coord, y_coord, time_coord: str
        Names of DataArrays in `dataset_in` to use as x, y, and time coordinates.
    bands: list-like
        A list-like containing 3 names of data variables in `dataset` to use as the red, green, and blue
        bands, respectively.
    min_possible, max_possible: int
        The minimum and maximum valid values for the selected bands according to
        the platform used to retrieve the data in `dataset`.
        For example, for Landsat these are generally 0 and 10000, respectively.
    use_data_min: bool
        Whether to use `min_possible` or the minimum among all selected bands
        as the band value which has a minimal intensity.
    use_data_max: bool
        Whether to use `max_possible` or the maximum among all selected bands
        as the band value which has a maximal intensity.
    min_inten, max_inten: float
        The min and max intensities for any band. These can be in range [0,1].
        These can be used to brighten or darken the image.
    width: int
        The width of the figure in inches.
    fig: matplotlib.figure.Figure
        The figure to use for the plot.
        If only `fig` is supplied, the Axes object used will be the first.
    ax: matplotlib.axes.Axes
        The axes to use for the plot.
    imshow_kwargs: dict
        The dictionary of keyword arguments passed to `ax.imshow()`.
        You can pass a colormap here with the key 'cmap'.

    Returns
    -------
    fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes used for the plot.
    """
    imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs

    ### < Dataset to RGB Format, needs float values between 0-1 
    rgb = np.stack([dataset[bands[0]],
                    dataset[bands[1]],
                    dataset[bands[2]]], axis = -1)
    # Interpolate values to be in the range [0,1] for creating the image.
    min_rgb = np.nanmin(rgb) if use_data_min else min_possible
    max_rgb = np.nanmax(rgb) if use_data_max else max_possible
    rgb = np.interp(rgb, (min_rgb, max_rgb), [min_inten,max_inten])
    rgb = rgb.astype(float)
    ### > 
    
    ### < takes a T/F mask, apply a color to T areas  
    for mask, color in paint_on_mask:        
        rgb[mask] = np.array(color)/ 255.0
    ### > 
    
    fig, ax = retrieve_or_create_fig_ax(fig, ax, figsize=figure_ratio(rgb.shape[:2], fixed_width = width))

    xarray_set_axes_labels(dataset, ax, x_coord, y_coord)
   
    if 'time' in dataset.dims:
        ax.imshow(rgb[at_index], **imshow_kwargs)
    else:
        ax.imshow(rgb, **imshow_kwargs)
    
    return fig, ax