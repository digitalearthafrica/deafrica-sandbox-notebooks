import folium
import itertools    
import math
import numpy as np
import pandas as pd
import datetime, time
from scipy.optimize import curve_fit
from scipy.signal import gaussian
from scipy import stats, exp
import warnings
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import CubicSpline
from . import dc_utilities as utilities
from matplotlib.colors import LinearSegmentedColormap
from copy import copy

## bin.py ##
def bin_time(dataset, bins):
    """
    Bins a dataset along the 'time' coordinate. The bins are equally spaced and the times for each 
    
    Parameters
    ----------
    dataset: xarray.Dataset
        
    bins: int
        The number of bins to use.
    """
    raise NotImplementedError("bin_time() has not yet been implemented")
    
## end bin.py ##

## Exporting ##

def time_to_string(t):
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(t.astype(int)/1000000000))

def export_slice_to_geotiff(ds, path):
    utilities.write_geotiff_from_xr(path, ds.astype(np.float32), list(ds.data_vars.keys()), crs="EPSG:4326")

def export_xarray_to_geotiff(ds, path):
    for t in ds.time:
        time_slice_xarray = ds.sel(time = t)
        export_slice_to_geotiff(time_slice_xarray, path + "_" + time_to_string(t) + ".tif")

## End exporting ##

## Compositing and masking ##
    
def restore_or_convert_dtypes(dtype_for_all, band_list, dataset_in_dtypes, dataset_out, no_data=-9999):
    """
    Restores original datatypes to data variables in Datasets
    output by mosaic functions.

    Parameters
    ----------
    dtype_for_all: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.
    band_list: list-like
        A list-like of the data variables in the dataset.
    dataset_in_dtypes: dict
        A dictionary mapping band names to datatypes.
    no_data: int or float
        The no data value.

    Returns
    -------
    dataset_out: xarray.Dataset
        The output Dataset.
    """
    if dtype_for_all is not None:
        # Integer types can't represent nan.
        if np.issubdtype(dtype_for_all, np.integer): # This also works for Python int type.
            utilities.nan_to_num(dataset_out, no_data)
        convert_to_dtype(dataset_out, dtype_for_all)
    else:  # Restore dtypes to state before masking.
        for band in band_list:
            band_dtype = dataset_in_dtypes[band]
            if np.issubdtype(band_dtype, np.integer):
                utilities.nan_to_num(dataset_out[band], no_data)
            dataset_out[band] = dataset_out[band].astype(band_dtype)
    return dataset_out

def landsat_clean_mask_invalid(dataset):
    """
    Masks out invalid data according to the LANDSAT 7 and 8 surface reflectance 
    specifications. See this document: 
    https://landsat.usgs.gov/sites/default/files/documents/ledaps_product_guide.pdf pages 19-20.
    
    Parameters
    ----------
    dataset: xarray.Dataset
        An xarray `Dataset` containing bands such as 'red', 'green', or 'blue'.
    """
    data_arr_names = [arr_name for arr_name in list(dataset.data_vars) if arr_name not in ['pixel_qa', 'radsat_qa', 'cloud_qa']]
    for data_arr_name in data_arr_names:
        dataset[data_arr_name] = dataset[data_arr_name].where((0 < dataset[data_arr_name]) & (dataset[data_arr_name] < 10000))
    return dataset

def create_default_clean_mask(dataset_in):
    """
    Description:
        Creates a data mask that masks nothing.
    -----
    Inputs:
        dataset_in (xarray.Dataset) - dataset retrieved from the Data Cube.
    Throws:
        ValueError - if dataset_in is an empty xarray.Dataset.
    """
    data_vars = dataset_in.data_vars
    if len(data_vars) != 0:
        first_data_var = next(iter(data_vars))
        clean_mask = np.ones(dataset_in[first_data_var].shape).astype(np.bool)
        return clean_mask
    else:
        raise ValueError('`dataset_in` has no data!')

def create_max_ndvi_mosaic(dataset_in, clean_mask=None, no_data=-9999, dtype=None, intermediate_product=None, **kwargs):
    """
    Method for calculating the pixel value for the max ndvi value.

    Parameters
    ----------
    dataset_in: xarray.Dataset
        A dataset retrieved from the Data Cube; should contain:
        coordinates: time, latitude, longitude
        variables: variables to be mosaicked (e.g. red, green, and blue bands)
    clean_mask: np.ndarray
        An ndarray of the same shape as `dataset_in` - specifying which values to mask out.
        If no clean mask is specified, then all values are kept during compositing.
    no_data: int or float
        The no data value.
    dtype: str or numpy.dtype
        A string denoting a Python datatype name (e.g. int, float) or a NumPy dtype (e.g.
        np.int16, np.float32) to convert the data to.

    Returns
    -------
    dataset_out: xarray.Dataset
        Compositited data with the format:
        coordinates: latitude, longitude
        variables: same as dataset_in
    """
    dataset_in = dataset_in.copy(deep=True)

    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)

    # Save dtypes because masking with Dataset.where() converts to float64.
    band_list = list(dataset_in.data_vars)
    dataset_in_dtypes = {}
    for band in band_list:
        dataset_in_dtypes[band] = dataset_in[band].dtype

    # Mask out clouds and scan lines.
    dataset_in = dataset_in.where((dataset_in != -9999) & clean_mask)

    if intermediate_product is not None:
        dataset_out = intermediate_product.copy(deep=True)
    else:
        dataset_out = None

    time_slices = range(len(dataset_in.time))
    for timeslice in time_slices:
        dataset_slice = dataset_in.isel(time=timeslice).drop('time')
        ndvi = (dataset_slice.nir - dataset_slice.red) / (dataset_slice.nir + dataset_slice.red)
        ndvi.values[np.invert(clean_mask)[timeslice, ::]] = -1000000000
        dataset_slice['ndvi'] = ndvi
        if dataset_out is None:
            dataset_out = dataset_slice.copy(deep=True)
            utilities.clear_attrs(dataset_out)
        else:
            for key in list(dataset_slice.data_vars):
                dataset_out[key].values[dataset_slice.ndvi.values > dataset_out.ndvi.values] = \
                    dataset_slice[key].values[dataset_slice.ndvi.values > dataset_out.ndvi.values]
    # Handle datatype conversions.
    dataset_out = restore_or_convert_dtypes(dtype, band_list, dataset_in_dtypes, dataset_out, no_data)
    return dataset_out

# End compositing and masking ##

def NDVI(dataset, normalize=True):
    """
    Computes the Normalized Difference Vegetation Index for an `xarray.Dataset`.
    Values should be in the range [-1,1] for valid LANDSAT data (nir and red are positive).
    
    Parameters
    ----------
    dataset: xarray.Dataset
        An `xarray.Dataset` that must contain 'nir' and 'red' `DataArrays`.
    normalize: bool
        Whether or not to normalize to the range [0,1].
    
    Returns
    -------
    ndvi: xarray.DataArray
        An `xarray.DataArray` with the same shape as `dataset` - the same coordinates in 
        the same order.
    """
    with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        ndvi = (dataset.nir - dataset.red) / (dataset.nir + dataset.red)
    if normalize:
        ndvi_min, ndvi_max = ndvi.min(), ndvi.max()
        ndvi = (ndvi - ndvi_min)/(ndvi_max - ndvi_min)
    return ndvi

def display_map(latitude = None, longitude = None, resolution = None):
    """ Generates a folium map with a lat-lon bounded rectangle drawn on it. Folium maps can be 
    
    Args:
        latitude   (float,float): a tuple of latitude bounds in (min,max) format
        longitude  ((float, float)): a tuple of longitude bounds in (min,max) format
        resolution ((float, float)): tuple in (lat,lon) format used to draw a grid on your map. Values denote   
                                     spacing of latitude and longitude lines.  Gridding starts at top left 
                                     corner. Default displays no grid at all.  

    Returns:
        folium.Map: A map centered on the lat lon bounds. A rectangle is drawn on this map detailing the
        perimeter of the lat,lon bounds.  A zoom level is calculated such that the resulting viewport is the
        closest it can possibly get to the centered bounding rectangle without clipping it. An 
        optional grid can be overlaid with primitive interpolation.  

    .. _Folium
        https://github.com/python-visualization/folium

    """
    def _degree_to_zoom_level(l1, l2, margin = 0.0):
        degree = abs(l1 - l2) * (1 + margin)
        zoom_level_int = 0
        if degree != 0:
            zoom_level_float = math.log(360/degree)/math.log(2)
            zoom_level_int = int(zoom_level_float)
        else:
            zoom_level_int = 18
        return zoom_level_int
    
    assert latitude is not None
    assert longitude is not None

    ###### ###### ######   CALC ZOOM LEVEL     ###### ###### ######

    margin = -0.5
    zoom_bias = 0
    
    lat_zoom_level = _degree_to_zoom_level(margin = margin, *latitude ) + zoom_bias
    lon_zoom_level = _degree_to_zoom_level(margin = margin, *longitude) + zoom_bias
    zoom_level = min(lat_zoom_level, lon_zoom_level) 

    ###### ###### ######   CENTER POINT        ###### ###### ######
    
    center = [np.mean(latitude), np.mean(longitude)]

    ###### ###### ######   CREATE MAP         ###### ###### ######
    
    map_hybrid = folium.Map(
        location=center,
        zoom_start=zoom_level, 
        tiles=" http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google"
    )
    
    ###### ###### ######   RESOLUTION GRID    ###### ###### ######
    
    if resolution is not None:
        res_lat, res_lon = resolution

        lats = np.arange(abs(res_lat), *latitude)
        lons = np.arange(abs(res_lon), *longitude)

        vertical_grid   = map(lambda x :([x[0][0],x[1]],[x[0][1],x[1]]),itertools.product([latitude],lons))
        horizontal_grid = map(lambda x :([x[1],x[0][0]],[x[1],x[0][1]]),itertools.product([longitude],lats))

        for segment in vertical_grid:
            folium.features.PolyLine(segment, color = 'white', opacity = 0.3).add_to(map_hybrid)    
        
        for segment in horizontal_grid:
            folium.features.PolyLine(segment, color = 'white', opacity = 0.3).add_to(map_hybrid)   
    
    ###### ###### ######     BOUNDING BOX     ###### ###### ######
    
    line_segments = [(latitude[0],longitude[0]),
                     (latitude[0],longitude[1]),
                     (latitude[1],longitude[1]),
                     (latitude[1],longitude[0]),
                     (latitude[0],longitude[0])
                    ]
    
    
    
    map_hybrid.add_child(
        folium.features.PolyLine(
            locations=line_segments,
            color='red',
            opacity=0.8)
    )

    map_hybrid.add_child(folium.features.LatLngPopup())        

    return map_hybrid

def xarray_time_series_plot(dataset, plot_descs, fig_params={'figsize':(18,12)}, scale_params={}, fig=None, ax=None, max_times_per_plot=None, show_legend=True):
    """
    Plot data variables in an xarray.Dataset together in one figure, with different plot types for each 
    (e.g. box-and-whisker plot, line plot, scatter plot), and optional curve fitting to means or medians along time.
    Handles data binned with xarray.Dataset methods resample() and groupby(). That is, it handles data binned along time
    or across years (e.g. by week of year).
    
    Parameters
    -----------
    dataset: xarray.Dataset 
        A Dataset containing some bands like NDVI or WOFS.
        The primary coordinate must be 'time'.
    plot_descs: dict
        Dictionary mapping names of DataArrays in the Dataset to plot to dictionaries mapping aggregation types (e.g. 'mean', 'median')
        to lists of dictionaries mapping plot types (e.g. 'line', 'box', 'scatter') to keyword arguments for plotting.
        
        Aggregation happens within time slices and can be many-to-many or many-to-one. Some plot types require many-to-many aggregation, and some other plot types require many-to-one aggregation.
        Aggregation types can be any of ['mean', 'median', 'none'], with 'none' performing no aggregation.
        
        Plot types can be any of ['scatter', 'line', 'gaussian', 'poly', 'cubic_spline', 'box'].
        The plot type 'poly' requires a 'degree' entry mapping to an integer in its dictionary of keyword arguments.
        
        Here is an example:
        {'ndvi':       {'mean': [{'line': {'color': 'forestgreen', 'alpha':alpha}}],
                        'none':  [{'box': {'boxprops': {'facecolor':'forestgreen', 'alpha':alpha}, 
                                                        'showfliers':False}}]}}
        This example will create a green line plot of the mean of the 'ndvi' band as well as a green box plot of the 'ndvi' band.
    fig_params: dict
        Figure parameters dictionary (e.g. {'figsize':(12,6)}). Used to create a Figure ``if fig is None and ax is None``.
        Note that in the case of multiple plots being created (see ``max_times_per_plot`` below), figsize will be the size
        of each plot - not the entire figure.
    scale_params: dict
        Currently not used.
        Dictionary mapping names of DataArrays to scaling methods (e.g. {'ndvi': 'std', 'wofs':'norm'}). 
        The options are ['std', 'norm']. The option 'std' standardizes. The option 'norm' normalizes (min-max scales). 
        Note that of these options, only normalizing guarantees that the y values will be in a fixed range - namely [0,1].
    fig: matplotlib.figure.Figure
        The figure to use for the plot. The figure must have at least one Axes object.
        You can use the code ``fig,ax = plt.subplots()`` to create a figure with an associated Axes object.
        The code ``fig = plt.figure()`` will not provide the Axes object.
        The Axes object used will be the first. This is ignored if ``max_times_per_plot`` is less than the number of times.
    ax: matplotlib.axes.Axes
        The axes to use for the plot. This is ignored if ``max_times_per_plot`` is less than the number of times.
    max_times_per_plot: int
        The maximum number of times per plot. If specified, one plot will be generated for each group 
        of this many times. The plots will be arranged in a grid.
    show_legend: bool
        Whether or not to show the legend.
        
    Raises
    ------
    ValueError:
        If an aggregation type is not possible for a plot type
    
    :Authors:
        John Rattz (john.c.rattz@ama-inc.com)
    """
    # Lists of plot types that can and cannot accept many-to-one aggregation for each time slice.
    plot_types_requiring_aggregation = ['line', 'gaussian', 'poly', 'cubic_spline']
    plot_types_handling_aggregation = ['scatter'] + plot_types_requiring_aggregation
    plot_types_not_handling_aggregation = ['box']
    all_plot_types = plot_types_requiring_aggregation + plot_types_handling_aggregation + plot_types_not_handling_aggregation
    
    # Aggregation types that aggregate all values for a given time to one value.
    many_to_one_agg_types = ['mean', 'median'] 
    # Aggregation types that aggregate to many values or do not aggregate.
    many_to_many_agg_types = ['none']
    all_agg_types = many_to_one_agg_types + many_to_many_agg_types
    
    
    # Determine how the data was aggregated, if at all.
    possible_time_agg_strs = ['week', 'weekofyear', 'month']
    time_agg_str = 'time'
    for possible_time_agg_str in possible_time_agg_strs:
        if possible_time_agg_str in list(dataset.coords):
            time_agg_str = possible_time_agg_str
            break
    # Make the data 2D - time and a stack of all other dimensions.
    non_time_dims = list(set(dataset.dims)-{time_agg_str})
    all_plotting_bands = list(plot_descs.keys())
    all_plotting_data = dataset[all_plotting_bands].stack(stacked_data=non_time_dims)
    all_times = all_plotting_data[time_agg_str].values
    # Mask out times for which no data variable to plot has any non-NaN data.
    nan_mask_data_vars = list(all_plotting_data[all_plotting_bands].notnull().data_vars.values())
    for i, data_var in enumerate(nan_mask_data_vars):
        time_nan_mask = data_var.values if i == 0 else time_nan_mask | data_var.values
    time_nan_mask = np.any(time_nan_mask, axis=1)
    times_not_all_nan = all_times[time_nan_mask]
    all_plotting_data = all_plotting_data.loc[{time_agg_str:times_not_all_nan}]
    
    # Scale
    if isinstance(scale_params, str): # if scale_params denotes the scaling type for the whole Dataset, scale the Dataset.
        all_plotting_data = xr_scale(all_plotting_data, scaling=scale_params)
    elif len(scale_params) > 0: # else, it is a dictionary denoting how to scale each DataArray.
        for data_arr_name, scaling in scale_params.items():
            all_plotting_data[data_arr_name] = xr_scale(all_plotting_data[data_arr_name], scaling=scaling)
    
    # Handle the potential for multiple plots.
    max_times_per_plot = len(times_not_all_nan) if max_times_per_plot is None else max_times_per_plot
    num_plots = int(np.ceil(len(times_not_all_nan)/max_times_per_plot))
    subset_num_cols = 2
    subset_num_rows = int(np.ceil(num_plots / subset_num_cols))
    if num_plots > 1:
        figsize = fig_params.pop('figsize')
        fig = plt.figure(figsize=figsize, **fig_params)
    
    # Create each plot.
    for time_ind, fig_ind in zip(range(0, len(times_not_all_nan), max_times_per_plot), range(num_plots)):
        lower_time_bound_ind, upper_time_bound_ind = time_ind, min(time_ind+max_times_per_plot, len(times_not_all_nan))
        time_extents = times_not_all_nan[[lower_time_bound_ind, upper_time_bound_ind-1]]
        # Retrieve or create the axes if necessary.
        if len(times_not_all_nan) <= max_times_per_plot:
            ax = retrieve_or_create_ax(fig, ax, **fig_params)
        else:
            ax = fig.add_subplot(subset_num_rows, subset_num_cols, fig_ind + 1)
        fig_times_not_all_nan = times_not_all_nan[lower_time_bound_ind:upper_time_bound_ind]
        plotting_data = all_plotting_data.loc[{time_agg_str:fig_times_not_all_nan}]
        epochs = np.array(list(map(n64_to_epoch, fig_times_not_all_nan))) if time_agg_str == 'time' else None
        x_locs = np_scale(epochs if time_agg_str == 'time' else fig_times_not_all_nan)
        
        # Data variable plots within each plot.
        data_arr_plots = []
        legend_labels = []
        # For each data array to plot...
        for data_arr_name, agg_dict in plot_descs.items():
            # For each aggregation type (e.g. 'mean', 'median')...
            for agg_type, plot_dicts in agg_dict.items():
                # For each plot for this aggregation type...
                for plot_dict in plot_dicts:
                    for plot_type, plot_kwargs in plot_dict.items():
                        assert plot_type in all_plot_types, \
                            r"For the '{}' DataArray: plot_type '{}' not recognized".format(data_arr_name, plot_type)
                        full_data_arr_plotting_data = plotting_data[data_arr_name].values
                        # Any times with all nan data are ignored in any plot type.
                        data_arr_nan_mask = np.any(~np.isnan(full_data_arr_plotting_data), axis=1)
            
                        # Skip plotting this data variable if it does not have enough data to plot.
                        if skip_plot(np.sum(data_arr_nan_mask), plot_type, plot_kwargs):
                            continue

                        # Remove times with all nan data.
                        data_arr_plotting_data = full_data_arr_plotting_data[data_arr_nan_mask]
                        # Large scales for x_locs can break the curve fitting for some reason.
                        data_arr_x_locs = x_locs[data_arr_nan_mask]
                        
                        # Some plot types require aggregation.
                        if plot_type in plot_types_requiring_aggregation:
                            if agg_type not in many_to_one_agg_types:
                                raise ValueError("For the '{}' DataArray: the plot type '{}' requires aggregation "
                                                 "(currently using '{}'). Please pass any of {} as the aggregation type "
                                                 "or change the plot type.".format(data_arr_name, plot_type, agg_type, many_to_one_agg_types))
                        # Some plot types cannot accept many-to-one aggregation.
                        if plot_type not in plot_types_handling_aggregation:
                            if agg_type not in many_to_many_agg_types:
                                raise ValueError("For the '{}' DataArray: the plot type '{}' doesn't accept aggregation "
                                                 "(currently using '{}'). Please pass any of {} as the aggregation type "
                                                 "or change the plot type.".format(data_arr_name, plot_type, agg_type, many_to_many_agg_types))
                        
                        if agg_type == 'mean':
                            y = ignore_warnings(np.nanmean, data_arr_plotting_data, axis=1)
                        elif agg_type == 'median':
                            y = ignore_warnings(np.nanmedian, data_arr_plotting_data, axis=1)
                        elif agg_type == 'none':
                            y = data_arr_plotting_data
            
                        # Create specified plot types.
                        plot_type_str = "" # Used to label the legend.
                        if plot_type == 'scatter':
                            # Ignore warning about taking the mean of an empty slice.        
                            data_arr_plots.append(ax.scatter(data_arr_x_locs, y, **plot_kwargs))
                            plot_type_str += 'scatterplot'
                        elif plot_type == 'line':
                            data_arr_plots.append(ax.plot(data_arr_x_locs, y, **plot_kwargs)[0])
                            plot_type_str += 'lineplot'
                        elif plot_type == 'box':
                            boxplot_nan_mask = ~np.isnan(y)
                            filtered_formatted_data = [] # Data formatted for matplotlib.pyplot.boxplot().
                            for i, (d, m) in enumerate(zip(y, boxplot_nan_mask)):
                                if len(d[m] != 0):
                                    filtered_formatted_data.append(d[m])
                            box_width = 0.5*np.min(np.diff(data_arr_x_locs)) if len(data_arr_x_locs) > 1 else 0.5
                            # Provide default arguments.
                            plot_kwargs.setdefault('boxprops', dict(facecolor='orange'))
                            plot_kwargs.setdefault('flierprops', dict(marker='o', markersize=0.5))
                            plot_kwargs.setdefault('showfliers', False)
                            bp = ax.boxplot(filtered_formatted_data, widths=[box_width]*len(filtered_formatted_data), 
                                            positions=data_arr_x_locs, patch_artist=True, 
                                            manage_xticks=False, **plot_kwargs) # `manage_xticks=False` to avoid excessive padding on the x-axis.
                            data_arr_plots.append(bp['boxes'][0])
                            plot_type_str += 'boxplot'
                        elif plot_type == 'gaussian':
                            data_arr_plots.append(plot_curvefit(data_arr_x_locs, y, fit_type=plot_type, plot_kwargs=plot_kwargs, ax=ax))
                            plot_type_str += 'gaussian fit'
                        elif plot_type == 'poly':
                            assert 'degree' in plot_kwargs, r"For the '{}' DataArray: When using 'poly' as the fit type," \
                                                            "the fit kwargs must have 'degree' specified.".format(data_arr_name)
                            data_arr_plots.append(plot_curvefit(data_arr_x_locs, y, fit_type=plot_type, plot_kwargs=plot_kwargs, ax=ax))
                            plot_type_str += 'degree {} polynomial fit'.format(plot_kwargs['degree'])
                        elif plot_type == 'cubic_spline':
                            data_arr_plots.append(plot_curvefit(data_arr_x_locs, y, fit_type=plot_type, plot_kwargs=plot_kwargs, ax=ax))
                            plot_type_str += 'cubic spline fit'
                        plot_type_str += ' of {}'.format(agg_type) if agg_type != 'none' else ''
                        legend_labels.append('{} of {}'.format(plot_type_str, data_arr_name))
                            

        # Label the axes and create the legend.
        date_strs = np.array(list(map(lambda time: np_dt64_to_str(time), fig_times_not_all_nan))) if time_agg_str=='time' else\
                    naive_months_ticks_by_week(fig_times_not_all_nan) if time_agg_str in ['week', 'weekofyear'] else\
                    month_ints_to_month_names(fig_times_not_all_nan)
        plt.xticks(x_locs, date_strs, rotation=45, ha='right', rotation_mode='anchor')
        if show_legend:
            plt.legend(handles=data_arr_plots, labels=legend_labels, loc='best')
        plt.title("Figure {}: Time Range {} to {}".format(fig_ind, date_strs[0], date_strs[-1]))
        plt.tight_layout()

def retrieve_or_create_ax(fig=None, ax=None, **fig_params):
    """
    Returns an appropriate Axes object given possible Figure or Axes objects.
    If neither is supplied, a new figure will be created with associated axes.
    """
    if fig is None:
        if ax is None:
            fig, ax = plt.subplots(**fig_params)
    else:
        ax = fig.axes[0]
    return ax

def n64_to_epoch(timestamp):
    ts = pd.to_datetime(str(timestamp)) 
    time_format = "%Y-%m-%d"
    ts = ts.strftime(time_format)
    epoch = int(time.mktime(time.strptime(ts, time_format)))
    return epoch

## Curve fitting ##

def plot_curvefit(x, y, fit_type, x_smooth=None, n_pts=200, fig_params={}, plot_kwargs={}, fig=None, ax=None):
    """
    Plots a curve fit given x values, y values, a type of curve to plot, and parameters for that curve.
    
    Parameters
    ----------
    x: np.ndarray
        A 1D NumPy array. The x values to fit to.
    y: np.ndarray
        A 1D NumPy array. The y values to fit to.
    fit_type: str
        The type of curve to fit. One of ['poly', 'gaussian', 'cubic_spline'].
        The option 'poly' plots a polynomial fit. The option 'gaussian' plots a Gaussian fit.
        The option 'cubic_spline' plots a cubic spline fit.
    x_smooth: list-like
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    fig_params: dict
        Figure parameters dictionary (e.g. {'figsize':(12,6)}).
        Used to create a Figure ``if fig is None and ax is None``.
    plot_kwargs: dict
        The kwargs for the call to ``matplotlib.axes.Axes.plot()``.
    fig: matplotlib.figure.Figure
        The figure to use for the plot. The figure must have at least one Axes object.
        You can use the code ``fig,ax = plt.subplots()`` to create a figure with an associated Axes object.
        The code ``fig = plt.figure()`` will not provide the Axes object. 
        The Axes object used will be the first.
    ax: matplotlib.axes.Axes
        The axes to use for the plot.
        
    Returns
    -------
    lines: matplotlib.lines.Line2D
        Can be used as a handle for a matplotlib legend (i.e. plt.legend(handles=...)) among other things.
    """
    # Avoid modifying the original arguments.
    fig_params = copy(fig_params)
    plot_kwargs = copy(plot_kwargs)
    
    fig_params.setdefault('figsize', (12,6))
    plot_kwargs.setdefault('linestyle', '-')

    # Retrieve or create the axes if necessary.
    ax = retrieve_or_create_ax(fig, ax, **fig_params)
    if x_smooth is None:
        x_smooth = np.linspace(x.min(), x.max(), n_pts)
    if fit_type == 'gaussian':
        y_smooth = gaussian_fit(x, y, x_smooth)
    elif fit_type == 'poly':
        assert 'degree' in plot_kwargs.keys(), "When plotting a polynomal fit, there must be" \
                                              "a 'degree' entry in the plot_kwargs parameter."
        degree = plot_kwargs.pop('degree')
        y_smooth = poly_fit(x, y, degree, x_smooth)
    elif fit_type == 'cubic_spline':
        cs = CubicSpline(x,y)
        y_smooth = cs(x_smooth)
    return ax.plot(x_smooth, y_smooth, **plot_kwargs)[0]

def gauss(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def gaussian_fit(x, y, x_smooth=None, n_pts=200):
    """
    Fits a Gaussian to some data - x and y. Returns predicted interpolation values.
    
    Parameters
    ----------
    x: list-like
        The x values of the data to fit to.
    y: list-like
        The y values of the data to fit to.
    x_smooth: list-like
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    """
    if x_smooth is None:
        x_smooth = np.linspace(x.min(), x.max(), n_pts)
    mean, sigma = np.nanmean(y), np.nanstd(y)
    popt,pcov = curve_fit(gauss,x,y,p0=[1,mean,sigma], maxfev=np.iinfo(np.int32).max)
    return gauss(x_smooth,*popt)
    
def poly_fit(x, y, degree, x_smooth=None, n_pts=200):
    """
    Fits a polynomial of any positive integer degree to some data - x and y. Returns predicted interpolation values.
    
    Parameters
    ----------
    x: list-like
        The x values of the data to fit to.
    y: list-like
        The y values of the data to fit to.
    x_smooth: list-like
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    degree: int
        The degree of the polynomial to fit.
    """
    if x_smooth is None:
        x_smooth = np.linspace(x.min(), x.max(), n_pts)
    return np.array([np.array([coef*(x_val**current_degree) for coef, current_degree in 
                               zip(np.polyfit(x, y, degree), range(degree, -1, -1))]).sum() for x_val in x_smooth])
    
## End curve fitting ##

def np_dt64_to_str(np_datetime, fmt='%Y-%m-%d'):
    """Converts a NumPy datetime64 object to a string based on a format string supplied to pandas strftime."""
    return pd.to_datetime(str(np_datetime)).strftime(fmt)

def tfmt(x, pos=None):
    return time.strftime("%Y-%m-%d",time.gmtime(x))

## Matplotlib colormap functions ##

def create_discrete_color_map(data_range, th, colors, cmap_name='my_cmap'):
    """
    Creates a discrete matplotlib LinearSegmentedColormap with thresholds for color changes.
    
    Parameters
    ----------
    data_range: list-like
        A 2-tuple of the minimum and maximum values the data may take.
    th: list
        Threshold values. Must be in the range of `data_range` - noninclusive.
    colors: list
        Colors to use between thresholds, so `len(colors) == len(th)+1`.
        Colors can be string names of matplotlib colors or 3-tuples of rgb values in range [0,255].
    cmap_name: str
        The name of the colormap for matplotlib.
    """
    import matplotlib as mpl
    # Normalize threshold values based on the data range.
    th = list(map(lambda val: (val - data_range[0])/(data_range[1] - data_range[0]), th))
    # Normalize color values.
    for i, color in enumerate(colors):
        if isinstance(color, tuple):
            colors[i] = [rgb/255 for rgb in color]
    th = [0.0] + th + [1.0]
    cdict = {}
    # These are fully-saturated red, green, and blue - not the matplotlib colors for 'red', 'green', and 'blue'.
    primary_colors = ['red', 'green', 'blue']
    # Get the 3-tuples of rgb values for the colors.
    color_rgbs = [(mpl.colors.to_rgb(color) if isinstance(color,str) else color) for color in colors]
    # For each color entry to go into the color dictionary...
    for primary_color_ind, primary_color in enumerate(primary_colors):
        cdict_entry = [None]*len(th)
        # For each threshold (as well as 0.0 and 1.0), specify the values for this primary color.
        for row_ind, th_ind in enumerate(range(len(th))):
            # Get the two colors that this threshold corresponds to.
            th_color_inds = [0,0] if th_ind==0 else \
                            [len(colors)-1, len(colors)-1] if th_ind==len(th)-1 else \
                            [th_ind-1, th_ind]
            primary_color_vals = [color_rgbs[th_color_ind][primary_color_ind] for th_color_ind in th_color_inds]
            cdict_entry[row_ind] = (th[th_ind],) + tuple(primary_color_vals)
        cdict[primary_color] = cdict_entry
    cmap = LinearSegmentedColormap(cmap_name, cdict)
    return cmap

## End matplotlib colormap functions ##

## Misc ##

def skip_plot(n_pts, plot_type, kwargs={}):
    """Returns a boolean denoting whether to skip plotting data given the number of points it contains."""
    min_pts_dict = {'scatter': 1, 'box': 1, 'gaussian': 3, 'poly': 1, 'cubic_spline': 3, 'line':2}
    min_pts = min_pts_dict[plot_type]
    if plot_type == 'poly':
        assert 'degree' in kwargs.keys(), "When plotting a polynomal fit, there must be" \
                                              "a 'degree' entry in the fit_kwargs parameter."
        degree = kwargs['degree']
        min_pts = min_pts + degree
    return n_pts < min_pts

def xarray_sortby_coord(dataset, coord):
    """
    Sort an xarray.Dataset by a coordinate. xarray.Dataset.sortby() sometimes fails, so this is an alternative.
    Credit to https://stackoverflow.com/a/42600594/5449970.
    """
    return dataset.loc[{coord:np.sort(dataset.coords[coord].values)}]

def xarray_values_in(data, values, data_vars=None):
    """
    Returns a mask for an xarray Dataset or DataArray, with `True` wherever the value is in values.
    
    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray
        The data to check for value matches.
    values: list-like
        The values to check for.
    data_vars: list-like
        The names of the data variables to check.
    
    Returns
    -------
    mask: np.ndarray
        A NumPy array shaped like ``data``. The mask can be used to mask ``data``.
        That is, ``data.where(mask)`` is an intended use.
    """
    if isinstance(data, xr.Dataset):
        mask = np.full_like(list(data.data_vars.values())[0], False, dtype=np.bool)
        for data_arr in data.data_vars.values():
            for value in values:
                mask = mask | (data_arr.values == value)
    elif isinstance(data, xr.DataArray):
        mask = np.full_like(data, False, dtype=np.bool)
        for value in values:
            mask = mask | (data.values == value)
    return mask

def ignore_warnings(func, *args, **kwargs):
    """Runs a function while ignoring warnings"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ret = func(*args, **kwargs)
    return ret

def xr_scale(data, data_vars=None, min_max=None, scaling='norm'):
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
    """
    data = data.copy()
    if isinstance(data, xr.Dataset):
        data_arr_names = list(data.data_vars) if data_vars is None else data_vars
        for data_arr_name in data_arr_names:
            data_arr = data[data_arr_name]
            data_arr.values = np_scale(data_arr.values, min_max=min_max, scaling=scaling)
    elif isinstance(data, xr.DataArray): 
        data.values = np_scale(data.values, min_max=min_max, scaling=scaling)
    return data
    
def np_scale(arr, pop_arr=None, pop_min_max=None, mean_std=None, min_max=None, scaling='norm'):
    """
    Scales a NumPy array with standard scaling or norm scaling.
    
    Parameters
    ----------
    arr: numpy.ndarray
        The NumPy array to scale.
    pop_arr: numpy.ndarray
        The NumPy array to treat as the population. 
        If specified, all members of arr must be within pop_arr or min_max must be specified.
    pop_min_max: tuple
        A 2-tuple of the population minimum and maximum, in that order. 
        Supercedes pop_arr when normalizing.
    mean_std: tuple
        A 2-tuple of the population mean and standard deviation, in that order. 
        Supercedes pop_arr when standard scaling.
    min_max: tuple
        A 2-tuple which specifies the desired range of the final output - the minimum and the maximum, in that order.
        If all values are the same, all values will become min_max[0].
    scaling: str
        The options are ['std', 'norm']. 
        The option 'std' standardizes. The option 'norm' normalizes (min-max scales). 
    """
    pop_arr = arr if pop_arr is None else pop_arr
    if scaling == 'norm':
        pop_min, pop_max = (pop_min_max[0], pop_min_max[1]) if pop_min_max is not None else (np.nanmin(pop_arr), np.nanmax(pop_arr))
        numerator, denominator = arr - pop_min, pop_max - pop_min
    elif scaling == 'std':
        mean, std = mean_std if mean_std is not None else (np.nanmean(pop_arr), np.nanstd(pop_arr))
        numerator, denominator = arr - mean, std
    # Primary scaling
    new_arr = arr
    if denominator > 0:
        new_arr = numerator / denominator
    # Optional final scaling.
    if min_max is not None:
        new_arr = np.interp(new_arr, (np.nanmin(new_arr), np.nanmax(new_arr)), min_max) if denominator > 0 else \
                  np.full_like(new_arr, min_max[0]) # The values are identical - set all values to the low end of the desired range.
    return new_arr

def remove_non_unique_ordered_list_str(ordered_list):
    """
    Sets all occurrences of a value in an ordered list after its first occurence to ''.
    For example, ['a', 'a', 'b', 'b', 'c'] would become ['a', '', 'b', '', 'c'].
    """
    prev_unique_str = ""
    for i in range(len(ordered_list)):
        current_str = ordered_list[i]
        if current_str != prev_unique_str:
            prev_unique_str = current_str
        else:
            ordered_list[i] = ""
    return ordered_list

# For February, assume leap years are included.
days_per_month = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 
                  7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

def get_weeks_per_month(num_weeks):
    """
    Including January, give 5 weeks to every third month - accounting for 
    variation between 52 and 54 weeks in a year by adding weeks to the last 3 months.
    """
    last_months_num_weeks = None
    if num_weeks <= 52:
        last_months_num_weeks = [5,4,4]
    elif num_weeks == 53:
        last_months_num_weeks = [5,4,5]
    elif num_weeks == 54:
        last_months_num_weeks = [5,5,5]
    return {month_int:num_weeks for (month_int,num_weeks) in zip(days_per_month.keys(), [5,4,4]*3+last_months_num_weeks)}

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def month_ints_to_month_names(month_ints):
    """
    Converts ordinal numbers for months (in range [1,12]) to their 3-letter names.
    """
    return [month_names[i-1] for i in month_ints]

def week_ints_to_month_names(week_ints):
    """
    Converts ordinal numbers for weeks (in range [1,54]) to their months' 3-letter names.
    """
    weeks_per_month = get_weeks_per_month(max(week_ints))
    week_month_strs = []
    for week_int in week_ints:
        month_int = -1
        for current_month_int, current_month_weeks in weeks_per_month.items():
            week_int -= current_month_weeks
            if week_int <= 0:
                month_int = current_month_int
                break
        week_month_strs.append(month_names[month_int-1])
    return week_month_strs

def naive_months_ticks_by_week(week_ints=None):
    """
    Given a list of week numbers (in range [1,54]), returns a list of month strings separated by spaces.
    Covers 54 weeks if no list-like of week numbers is given.
    This is only intended to be used for labeling axes in plotting.
    """
    month_ticks_by_week = []
    if week_ints is None: # Give month ticks for all weeks.
        month_ticks_by_week = week_ints_to_month_names(list(range(54)))
    else:
        month_ticks_by_week = remove_non_unique_ordered_list_str(week_ints_to_month_names(week_ints))
    return month_ticks_by_week