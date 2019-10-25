## deafrica_plotting.py
'''
Description: This file contains a set of python functions for plotting 
Digital Earth Africa data.

License: The code in this notebook is licensed under the Apache License, 
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Digital Earth 
Africa data is licensed under the Creative Commons by Attribution 4.0 
license (https://creativecommons.org/licenses/by/4.0/).

Contact: If you need assistance, please post a question on the Open Data
Cube Slack channel (http://slack.opendatacube.org/) or on the GIS Stack 
Exchange (https://gis.stackexchange.com/questions/ask?tags=open-data-cube) 
using the `open-data-cube` tag (you can view previously asked questions 
here: https://gis.stackexchange.com/questions/tagged/open-data-cube). 

If you would like to report an issue with this script, file one on 
Github: https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues/new

Functions included:
    rgb
    display_map
    map_shapefile
    animated_timeseries
    animated_timeseriesline
    animated_doubletimeseries

Last modified: October 2019

'''

# Import required packages
import folium  
import math
import numpy as np
import ipywidgets
import matplotlib as mpl
from pyproj import Proj, transform
from IPython.display import display
from ipyleaflet import Map, Marker, Popup, GeoJSON, basemaps
from skimage import exposure
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datetime import datetime
import calendar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def rgb(ds,
        bands=['red', 'green', 'blue'],
        index=None,
        index_dim='time',
        robust=True,
        percentile_stretch=None,
        col_wrap=4,
        size=6,
        aspect=None,
        savefig_path=None,
        savefig_kwargs={},
        **kwargs):
    
    """
    Takes an xarray dataset and plots RGB images using three imagery 
    bands (e.g ['nbart_red', 'nbart_green', 'nbart_blue']). The `index` 
    parameter allows easily selecting individual or multiple images for 
    RGB plotting. Images can be saved to file by specifying an output 
    path using `savefig_path`.
    
    This function was designed to work as an easier-to-use wrapper 
    around xarray's `.plot.imshow()` functionality.
    
    Last modified: October 2019
    
    Parameters
    ----------  
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array to plot as an RGB 
        image. If the array has more than two dimensions (e.g. multiple 
        observations along a 'time' dimension), either use `index` to 
        select one (`index=0`) or multiple observations 
        (`index=[0, 1]`), or create a custom faceted plot using e.g. 
        `col="time"`.       
    bands : list of strings, optional
        A list of three strings giving the band names to plot. Defaults 
        to '['red', 'green', 'blue']'.
    index : integer or list of integers, optional
        `index` can be used to select one (`index=0`) or multiple 
        observations (`index=[0, 1]`) from the input dataset for 
        plotting. If multiple images are requested these will be plotted
        as a faceted plot.
    index_dim : string, optional
        The dimension along which observations should be plotted if 
        multiple observations are requested using `index`. Defaults to 
        `time`.
    robust : bool, optional
        Produces an enhanced image where the colormap range is computed 
        with 2nd and 98th percentiles instead of the extreme values. 
        Defaults to True.
    percentile_stretch : tuple of floats
        An tuple of two floats (between 0.00 and 1.00) that can be used 
        to clip the colormap range to manually specified percentiles to 
        get more control over the brightness and contrast of the image. 
        The default is None; '(0.02, 0.98)' is equivelent to 
        `robust=True`. If this parameter is used, `robust` will have no 
        effect.
    col_wrap : integer, optional
        The number of columns allowed in faceted plots. Defaults to 4.
    size : integer, optional
        The height (in inches) of each plot. Defaults to 6.
    aspect : integer, optional
        Aspect ratio of each facet in the plot, so that aspect * size 
        gives width of each facet in inches. Defaults to None, which 
        will calculate the aspect based on the x and y dimensions of 
        the input data.
    savefig_path : string, optional
        Path to export image file for the RGB plot. Defaults to None, 
        which does not export an image file.
    savefig_kwargs : dict, optional
        A dict of keyword arguments to pass to 
        `matplotlib.pyplot.savefig` when exporting an image file. For 
        all available options, see: 
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html        
    **kwargs : optional
        Additional keyword arguments to pass to `xarray.plot.imshow()`. 
        For more options, see:
        http://xarray.pydata.org/en/stable/generated/xarray.plot.imshow.html  
        
    Returns
    -------
    An RGB plot of one or multiple observations, and optionally an image
    file written to file.
    
    """

    # Compute image aspect based on the last two dimensions (this will 
    # exclude the index dim if it is present in the dataset)
    if not aspect:
        x_dim, y_dim = list(ds.dims)[-2:]
        aspect = len(ds[x_dim]) / len(ds[y_dim])

    # If no value is supplied for `index` (the default), plot using default 
    # values and arguments passed via `**kwargs`
    if index is None:
        
        # Select bands and convert to DataArray
        da = ds[bands].to_array()

        # If percentile_stretch == True, clip plotting to percentile vmin, vmax
        if percentile_stretch:
            vmin, vmax = da.compute().quantile(percentile_stretch).values
            kwargs.update({'vmin': vmin, 'vmax': vmax})        
        
        # If there are more than three dimensions and the index dimension == 1, 
        # squeeze this dimension out to remove it
        if ((len(ds.dims) > 2) and 
            ('col' not in kwargs) and 
            (len(da[index_dim]) == 1)):
        
            da = da.squeeze(dim=index_dim)
            
        # If there are more than three dimensions and the index dimension
        # is longer than 1, raise exception to tell user to use 'col'/`index`
        elif ((len(ds.dims) > 2) and 
              ('col' not in kwargs) and 
              (len(da[index_dim]) > 1)):
                
            raise Exception(
                f'The input dataset `ds` has more than two dimensions: '
                '{list(ds.dims.keys())}. Please select a single observation '
                'using e.g. `index=0`, or enable faceted plotting by adding '
                'the arguments e.g. `col="time", col_wrap=4` to the function ' 
                'call'
            )

        img = da.plot.imshow(robust=robust,
                             col_wrap=col_wrap,
                             size=size,
                             aspect=aspect,
                             **kwargs)

    # If values provided for `index`, extract corresponding observations and 
    # plot as either single image or facet plot
    else:

        # If a float is supplied instead of an integer index, raise exception
        if isinstance(index, float):
            raise Exception(
                f'Please supply `index` as either an integer or a list of '
                'integers'
            )

        # If col argument is supplied as well as `index`, raise exception
        if 'col' in kwargs:
            raise Exception(
                f'Cannot supply both `index` and `col`; please remove one and '
                'try again'
            )

        # Convert index to generic type list so that number of indices supplied
        # can be computed
        index = index if isinstance(index, list) else [index]

        # Select bands and observations and convert to DataArray
        da = ds[bands].isel(**{index_dim: index}).to_array()

        # If percentile_stretch == True, clip plotting to percentile vmin, vmax
        if percentile_stretch:
            vmin, vmax = da.compute().quantile(percentile_stretch).values
            kwargs.update({'vmin': vmin, 'vmax': vmax})

        # If multiple index values are supplied, plot as a faceted plot
        if len(index) > 1:

            img = da.plot.imshow(robust=robust,
                                 col=index_dim,
                                 col_wrap=col_wrap,
                                 size=size,
                                 aspect=aspect,
                                 **kwargs)

        # If only one index is supplied, squeeze out index_dim and plot as a 
        # single panel
        else:

            img = da.squeeze(dim=index_dim).plot.imshow(robust=robust,
                                                        size=size,
                                                        aspect=aspect,
                                                        **kwargs)

    # If an export path is provided, save image to file. Individual and 
    # faceted plots have a different API (figure vs fig) so we get around this 
    # using a try statement:
    if savefig_path:

        print(f'Exporting image to {savefig_path}')

        try:
            img.fig.savefig(savefig_path, **savefig_kwargs)
        except:
            img.figure.savefig(savefig_path, **savefig_kwargs)

            
def display_map(x, y, crs='EPSG:4326', margin=-0.5, zoom_bias=0):
    """ 
    Given a set of x and y coordinates, this function generates an 
    interactive map with a bounded rectangle overlayed on Google Maps 
    imagery.        
    
    Last modified: September 2019
    
    Modified from function written by Otto Wagner available here: 
    https://github.com/ceos-seo/data_cube_utilities/tree/master/data_cube_utilities
    
    Parameters
    ----------  
    x : (float, float)
        A tuple of x coordinates in (min, max) format. 
    y : (float, float)
        A tuple of y coordinates in (min, max) format.
    crs : string, optional
        A string giving the EPSG CRS code of the supplied coordinates. 
        The default is 'EPSG:4326'.
    margin : float
        A numeric value giving the number of degrees lat-long to pad 
        the edges of the rectangular overlay polygon. A larger value 
        results more space between the edge of the plot and the sides 
        of the polygon. Defaults to -0.5.
    zoom_bias : float or int
        A numeric value allowing you to increase or decrease the zoom 
        level by one step. Defaults to 0; set to greater than 0 to zoom 
        in, and less than 0 to zoom out.
        
    Returns
    -------
    folium.Map : A map centered on the supplied coordinate bounds. A 
    rectangle is drawn on this map detailing the perimeter of the x, y 
    bounds.  A zoom level is calculated such that the resulting 
    viewport is the closest it can possibly get to the centered 
    bounding rectangle without clipping it. 
    """

    # Convert each corner coordinates to lat-lon
    all_x = (x[0], x[1], x[0], x[1])
    all_y = (y[0], y[0], y[1], y[1])
    all_longitude, all_latitude = transform(Proj(init=crs),
                                            Proj(init='EPSG:4326'), 
                                            all_x, all_y)

    # Calculate zoom level based on coordinates
    lat_zoom_level = _degree_to_zoom_level(min(all_latitude),
                                           max(all_latitude),
                                           margin=margin) + zoom_bias
    lon_zoom_level = _degree_to_zoom_level(min(all_longitude), 
                                           max(all_longitude), 
                                           margin=margin) + zoom_bias
    zoom_level = min(lat_zoom_level, lon_zoom_level)

    # Identify centre point for plotting
    center = [np.mean(all_latitude), np.mean(all_longitude)]

    # Create map
    interactive_map = folium.Map(
        location=center,
        zoom_start=zoom_level,
        tiles="http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google")

    # Create bounding box coordinates to overlay on map
    line_segments = [(all_latitude[0], all_longitude[0]),
                     (all_latitude[1], all_longitude[1]),
                     (all_latitude[3], all_longitude[3]),
                     (all_latitude[2], all_longitude[2]),
                     (all_latitude[0], all_longitude[0])]

    # Add bounding box as an overlay
    interactive_map.add_child(
        folium.features.PolyLine(locations=line_segments,
                                 color='red',
                                 opacity=0.8))

    # Add clickable lat-lon popup box
    interactive_map.add_child(folium.features.LatLngPopup())

    return interactive_map


def map_shapefile(gdf, 
                  weight=2, 
                  colormap=mpl.cm.YlOrRd, 
                  basemap=basemaps.Esri.WorldImagery, 
                  default_zoom=13,
                  hover_col=None,
                  hover_prefix=''):
    
    """
    Plots a geopandas GeoDataFrame over an interactive ipyleaflet 
    basemap. Optionally, can be set up to print selected data from 
    features in the GeoDataFrame. 
    
    Last modified: October 2019
    
    Parameters
    ----------  
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the spatial features to be plotted 
        over the basemap
    weight : float or int, optional
        An optional numeric value giving the weight that line features
        will be plotted as. Defaults to 2; larger numbers = thicker
    colormap : matplotlib.cm, optional
        An optional matplotlib.cm colormap used to style the features
        in the GeoDataFrame. Features will be coloured by the order
        they appear in the GeoDataFrame. Defaults to the `YlOrRd` 
        colormap.
    basemap : ipyleaflet.basemaps object, optional
        An optional ipyleaflet.basemaps object used as the basemap for 
        the interactive plot. Defaults to `basemaps.Esri.WorldImagery`
    default_zoom : int, optional
        An optional integer giving a default zoom level for the 
        interactive ipyleaflet plot. Defaults to 13
    hover_col : str, optional
        An optional string giving the name of any column in the
        GeoDataFrame you wish to have data from printed above the 
        interactive map when a user hovers over the features in the map.
        Defaults to None which will not print any data. 

    """
    
    def n_colors(n, colormap=colormap):
        data = np.linspace(0.0,1.0,n)
        c = [mpl.colors.rgb2hex(d[0:3]) for d in colormap(data)]
        return c

    def data_to_colors(data, colormap=colormap):
        c = [mpl.colors.rgb2hex(d[0:3]) for 
             d in colormap(mpl.colors.Normalize()(data))]
        return c 
    
    def on_hover(event, id, properties):
        with dbg:
            text = properties.get(hover_col, '???')
            lbl.value = f'{hover_col}: {text}'
            # print(properties)
  
    # Convert to WGS 84 and GeoJSON format
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    data = gdf_wgs84.__geo_interface__    
    
    # For each feature in dataset, append colour values
    n_features = len(data['features'])
    colors = n_colors(n_features)
    
    for feature, color in zip(data['features'], colors):
        feature['properties']['style'] = {'color': color, 
                                          'weight': weight, 
                                          'fillColor': color, 
                                          'fillOpacity': 1.0}

    # Get centroid to focus map on
    lon, lat = gdf_wgs84.unary_union.centroid.coords.xy 
    
    # Plot map 
    m = Map(center=(lat[0], lon[0]), 
            zoom=default_zoom, 
            basemap=basemap, 
            layout=dict(width='800px', height='600px'))
    
    # Add GeoJSON layer to map
    feature_layer = GeoJSON(data=data)
    m.add_layer(feature_layer)
    
    # If a column is specified by `hover_col`, print data from the
    # hovered feature above the map
    if hover_col:        
        lbl = ipywidgets.Label()
        dbg = ipywidgets.Output()        
        feature_layer.on_hover(on_hover)
        display(lbl)
      
    # Display the map
    display(m)

    
def animated_timeseries(ds,
                        output_path,
                        width_pixels=500,
                        interval=200,
                        bands=['red', 'green', 'blue'],
                        percentile_stretch=(0.02, 0.98),
                        image_proc_func=None,
                        title=False,
                        show_date=True,
                        annotation_kwargs={},
                        onebandplot_cbar=True,
                        onebandplot_kwargs={},
                        shapefile_path=None,
                        shapefile_kwargs={},
                        time_dim='time',
                        x_dim='x',
                        y_dim='y'):
    """
    Takes an xarray time series and animates the data as either a 
    three-band (e.g. true or false colour) or single-band animation, 
    allowing changes in the landscape to be compared across time.
    
    Animations can be exported as .mp4 (ideal for Twitter/social media)
    and .gif (ideal for all purposes, but can have large file sizes) 
    format files, and customised to include titles and date annotations 
    or use specific combinations of input bands.
    
    A shapefile boundary can be added to the output animation by 
    providing a path to the shapefile.
    
    This function can be used to produce visually appealing 
    cloud-free animations when used in combination with the `load_ard` 
    function from `deafrica-sandbox-notebooks/Scripts/deafrica_datahandling`.
    
    Last modified: October 2019
    
    Parameters
    ----------  
    ds : xarray.Dataset
        An xarray dataset with multiple time steps (i.e. multiple 
        observations along the `time` dimension).        
    output_path : str
        A string giving the output location and filename of the 
        resulting animation. File extensions of '.mp4' and '.gif' are 
        accepted.    
    width_pixels : int, optional
        An integer defining the output width in pixels for the resulting 
        animation. The height of the animation is set automatically 
        based on the dimensions/ratio of the input xarray dataset. 
        Defaults to 500 pixels wide.        
    interval : int, optional
        An integer defining the milliseconds between each animation 
        frame used to control the speed of the output animation. Higher 
        values result in a slower animation. Defaults to 200 
        milliseconds between each frame.         
    bands : list of strings, optional
        An optional list of either one or three bands to be plotted, 
        all of which must exist in `ds`. Defaults to 
        `['red', 'green', 'blue']`.         
    percentile_stretch : tuple of floats, optional
        An optional tuple of two floats that can be used to clip one or 
        three-band arrays by percentiles to produce a more vibrant, 
        visually attractive image that is not affected by outliers/
        extreme values. The default is `(0.02, 0.98)` which is 
        equivalent to xarray's `robust=True`.        
    image_proc_func : func, optional
        An optional function can be passed to modify three-band arrays 
        for each timestep prior to animating. This could include image 
        processing functions such as increasing contrast, unsharp 
        masking, saturation etc. The function should take AND return a 
        three-band numpy array with shape [:, :, 3]. If your function 
        has parameters, you can pass in custom values using `partial` 
        from `functools`: 
        `image_proc_func=partial(custom_func, param1=10)`.
    title : str or list of strings, optional
        An optional string or list of strings with a length equal to the
        number of timesteps in ds. This can be used to display a static 
        title (using a string), or a dynamic title (using a list) that 
        displays different text for each timestep. Defaults to False, 
        which plots no title.        
    show_date : bool, optional
        An optional boolean that defines whether or not to plot date 
        annotations for each animation frame. Defaults to True, which 
        plots date annotations based on ds.        
    annotation_kwargs : dict, optional
        An optional dict of kwargs for controlling the appearance of 
        text annotations to pass to the matplotlib `plt.annotate` 
        function (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html 
        for options). For example, `annotation_kwargs={'fontsize':20, 
        'color':'red', 'family':'serif'}. By default, text annotations 
        are plotted as white, size 20 mono-spaced font with a 2.5pt 
        black outline in the top-right of the animation.         
    onebandplot_cbar : bool, iptional
        An optional boolean indicating whether to include a colourbar 
        for one-band arrays. Defaults to True.        
    onebandplot_kwargs : dict, optional
        An optional dict of kwargs for controlling the appearance of 
        one-band image arrays to pass to matplotlib `plt.imshow` 
        (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html 
        for options). This only applies if an xarray with a single band 
        is passed to `ds`. For example, a green colour scheme and custom 
        stretch could be specified using: 
        `onebandplot_kwargs={'cmap':'Greens`, 'vmin':0.2, 'vmax':0.9}`. 
        By default, one-band arrays are plotted using the 'Greys' cmap 
        with bilinear interpolation.
        
        Two special kwargs (`tick_fontsize`, `tick_colour`) can also be 
        passed to control the tick labels on the colourbar. This can be 
        useful for example when the tick labels are difficult to see 
        against a dark background.       
    shapefile_path : str or list of strings, optional
        An optional string or list of strings giving the file paths of 
        one or multiple shapefiles to overlay on the output animation. 
        The shapefiles must be in the same projection as the input 
        xarray dataset.        
    shapefile_kwargs : dict or list of dicts, optional
        An optional dictionary of kwargs or list of dictionaries to 
        specify the appearance of the shapefile overlay by passing to 
        `GeoSeries.plot` (see http://geopandas.org/reference.html#geopandas.GeoSeries.plot). 
        For example: `shapefile_kwargs = {'linewidth':2, 
        'edgecolor':'black', 'facecolor':"#00000000"}`. If multiple 
        shapefiles were provided to `shapefile_path`, each shapefile can 
        be plotted with a different colour style by passing in a list of
        kwarg dicts of the same length as `shapefile_path`.        
    time_dim : str, optional
        An optional string allowing you to override the xarray dimension 
        used for time. Defaults to 'time'.
    x_dim : str, optional
        An optional string allowing you to override the xarray dimension 
        used for x coordinates. Defaults to 'x'.    
    y_dim : str, optional
        An optional string allowing you to override the xarray dimension 
        used for y coordinates. Defaults to 'y'.
        
    """

    ###############
    # Setup steps #
    ###############

    # Test if all dimensions exist in dataset
    if time_dim in ds and x_dim in ds and y_dim in ds:

        # First test if there are three bands, and that all exist in both datasets:
        if ((len(bands) == 3) |
            (len(bands) == 1)) & all([(b in ds.data_vars) for b in bands]):

            # Import xarrays as lists of three band numpy arrays
            imagelist, vmin, vmax = _ds_to_arrraylist(
                ds,
                bands=bands,
                time_dim=time_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                percentile_stretch=percentile_stretch,
                image_proc_func=image_proc_func)

            # Get time, x and y dimensions of dataset and calculate 
            # width vs height of plot
            timesteps = len(ds[time_dim])
            width = len(ds[x_dim])
            height = len(ds[y_dim])
            scale = (width_pixels / width)

            # If title is supplied as a string, multiply out to a list 
            # with one string per timestep. Otherwise, use supplied list
            # for plot titles.
            if isinstance(title, str) or isinstance(title, bool):
                title_list = [title] * timesteps
            else:
                title_list = title

            # Set up annotation parameters that plt.imshow plotting for 
            # single band array images. The nested dict structure sets 
            # default values which can be overwritten/customised by the
            # manually specified `onebandplot_kwargs`
            onebandplot_kwargs = dict({'cmap': 'Greys',
                                       'interpolation': 'bilinear',
                                       'vmin': vmin,
                                       'vmax': vmax,
                                       'tick_colour': 'black',
                                       'tick_fontsize': 12}, 
                                      **onebandplot_kwargs)

            # Use pop to remove the two special tick kwargs from the 
            # onebandplot_kwargs dict, and save individually
            onebandplot_tick_colour = onebandplot_kwargs.pop('tick_colour')
            onebandplot_tick_fontsize = onebandplot_kwargs.pop('tick_fontsize')

            # Set up annotation parameters that control font etc. The 
            # nested dict structure sets default values which can be 
            # overwritten/customised by the manually specified 
            #`annotation_kwargs`
            annotation_kwargs = dict(
                {
                    'xy': (1, 1),
                    'xycoords': 'axes fraction',
                    'xytext': (-5, -5),
                    'textcoords': 'offset points',
                    'horizontalalignment': 'right',
                    'verticalalignment': 'top',
                    'fontsize': 20,
                    'color': 'white',
                    'path_effects': [PathEffects.withStroke(linewidth=2.5, 
                                                            foreground='black')]
                }, **annotation_kwargs)

            ###################
            # Initialise plot #
            ###################

            # Set up figure
            fig, ax1 = plt.subplots(ncols=1)
            fig.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0,
                                hspace=0)
            
            fig.set_size_inches(width * scale / 72, 
                                height * scale / 72, 
                                forward=True)
            ax1.axis('off')

            # Initialise axesimage objects to be updated during 
            # animation, setting extent from dims
            extents = [float(ds[x_dim].min()),
                       float(ds[x_dim].max()),
                       float(ds[y_dim].min()),
                       float(ds[y_dim].max())]
            
            im = ax1.imshow(imagelist[0], 
                            extent=extents, 
                            **onebandplot_kwargs)

            # Initialise annotation objects to be updated during 
            # animation
            t = ax1.annotate('', **annotation_kwargs)

            #########################
            # Add optional overlays #
            #########################

            # Optionally add shapefile overlay(s) from either string 
            # path or list of string paths
            if isinstance(shapefile_path, str):

                # Define default plotting parameters for the overlaying 
                # shapefile(s). The nested dict structure sets default 
                # values which can be overwritten/customised by the 
                # manually specified `shapefile_kwargs`
                shapefile_kwargs = dict({'linewidth': 2,
                                         'edgecolor': 'black',
                                         'facecolor': "#00000000"}, 
                                        **shapefile_kwargs)

                shapefile = gpd.read_file(shapefile_path)
                shapefile.plot(**shapefile_kwargs, ax=ax1)

            elif isinstance(shapefile_path, list):

                # Iterate through list of string paths
                for i, shapefile in enumerate(shapefile_path):

                    if isinstance(shapefile_kwargs, list):

                        # If a list of shapefile_kwargs is supplied, use
                        # one for each shapefile
                        shapefile_kwargs_i = dict({'linewidth': 2,
                                                   'edgecolor': 'black',
                                                   'facecolor': "#00000000"}, 
                                                  **shapefile_kwargs[i])

                        shapefile = gpd.read_file(shapefile)
                        shapefile.plot(**shapefile_kwargs_i, ax=ax1)

                    else:

                        # If one shapefile_kwargs is provided, use for 
                        # all shapefiles
                        shapefile_kwargs = dict({'linewidth': 2,
                                                 'edgecolor': 'black',
                                                 'facecolor': "#00000000"}, 
                                                **shapefile_kwargs)

                        shapefile = gpd.read_file(shapefile)
                        shapefile.plot(**shapefile_kwargs, ax=ax1)

            # After adding shapefile, fix extents of plot
            ax1.set_xlim(extents[0], extents[1])
            ax1.set_ylim(extents[2], extents[3])

            # Optionally add colourbar for one band images
            if (len(bands) == 1) & onebandplot_cbar:

                _add_colourbar(ax1,
                               im,
                               tick_fontsize=onebandplot_tick_fontsize,
                               tick_colour=onebandplot_tick_colour,
                               vmin=onebandplot_kwargs['vmin'],
                               vmax=onebandplot_kwargs['vmax'],
                               cmap=onebandplot_kwargs['cmap'])

            ########################################
            # Create function to update each frame #
            ########################################

            # Function to update figure
            def update_figure(frame_i):

                # If possible, extract dates from time dimension
                try:

                    # Get human-readable date info (e.g. "16 May 1990")
                    ts = ds[time_dim][{time_dim: frame_i}].dt
                    year = ts.year.item()
                    month = ts.month.item()
                    day = ts.day.item()
                    date_string = '{} {} {}'.format(day,
                                                    calendar.month_abbr[month],
                                                    year)

                except:

                    date_string = ds[time_dim][{time_dim: frame_i}].values.item()

                # Create annotation string based on title and date 
                # specifications:
                title = title_list[frame_i]
                if title and show_date:
                    title_date = '{}\n{}'.format(date_string, title)
                elif title and not show_date:
                    title_date = '{}'.format(title)
                elif show_date and not title:
                    title_date = '{}'.format(date_string)
                else:
                    title_date = ''

                # Update figure for frame
                im.set_array(imagelist[frame_i])
                t.set_text(title_date)

                # Return the artists set
                return [im, t]

            ##############################
            # Generate and run animation #
            ##############################

            # Generate animation
            print('Generating {} frame animation'.format(timesteps))
            ani = animation.FuncAnimation(fig,
                                          update_figure,
                                          frames=timesteps,
                                          interval=interval,
                                          blit=True)

            # Export as either MP4 or GIF
            if output_path[-3:] == 'mp4':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path, dpi=72)

            elif output_path[-3:] == 'gif':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path,
                         writer='pillow') 

            else:
                print('    Output file type must be either .mp4 or .gif')

        else:
            print(
                'Please select either one or three bands that all exist in the input dataset'
            )

    else:
        print('At least one x, y or time dimension does not exist in the input dataset. Please use the `time_dim`,' \
              '`x_dim` or `y_dim` parameters to override the default dimension names used for plotting')

def animated_timeseriesline(ds, df, output_path, 
                            width_pixels=1000, interval=200, bands=['red', 'green', 'blue'],
                            percentile_stretch = (0.02, 0.98), image_proc_func=None,
                            title=False, show_date=True, annotation_kwargs={},
                            onebandplot_cbar=True, onebandplot_kwargs={}, 
                            shapefile_path=None, shapefile_kwargs={}, pandasplot_kwargs={},
                            time_dim = 'time', x_dim = 'x', y_dim = 'y'):
    
    """
    Takes an xarray time series and a pandas dataframe, and animates a line graph showing change in a variable 
    across time in the right column at the same time as a three-band (e.g. true or false colour) or single-band 
    animation in the left column.
    
    Animations can be exported as .mp4 (ideal for Twitter/social media), .wmv (ideal for Powerpoint) and .gif 
    (ideal for all purposes, but can have large file sizes) format files, and customised to include titles and 
    date annotations or use specific combinations of input bands. 
    A shapefile boundary can be added to the output animation by providing a path to the shapefile.
    
    This function can be used to produce visually appealing cloud-free animations when used in combination with 
    the `load_clearlandsat` function from `dea-notebooks/10_Scripts/DEADataHandling`.
    
    Last modified: October 2018
    Author: Robbi Bishop-Taylor, Sean Chua, Bex Dunn    
    
    :param ds: 
        An xarray dataset with multiple time steps (i.e. multiple observations along the `time` dimension) to plot 
        in the left panel of the animation.
        
    :param df: 
        An pandas dataframe with time steps contained in a DatetimeIndex column, and one or more numeric data 
        columns to plot as lines in the right panel. Column names are used to label the lines on the plot, so
        assign them informative names. Lines are plotted by showing all parts of the line with dates on or before
        the current timestep (i.e. for a 2006 time step, only the portion of the lines with dates on or before 
        2006 will be plotted for that frame.
        
    :param output_path: 
        A string giving the output location and filename of the resulting animation. File extensions of '.mp4', 
        '.wmv' and '.gif' are accepted.
    
    :param width_pixels:
        An integer defining the output width in pixels for the resulting animation. The height of the animation is
        set automatically based on the dimensions/ratio of the input xarray dataset. Defaults to 1000 pixels wide.
        
    :param interval:
        An integer defining the milliseconds between each animation frame used to control the speed of the output
        animation. Higher values result in a slower animation. Defaults to 200 milliseconds between each frame. 
        
    :param bands:
        An optional list of either one or three bands to be plotted in the left panel, all of which must exist in 
        `ds`. Defaults to `['red', 'green', 'blue']`. 
        
    :param percentile_stretch:
        An optional tuple of two floats that can be used to clip one or three-band arrays in the left panel by 
        percentiles to produce a more vibrant, visually attractive image that is not affected by outliers/extreme 
        values. The default is `(0.02, 0.98)` which is equivalent to xarray's `robust=True`.
        
    :param image_proc_func:
        An optional function can be passed to modify three-band arrays for each timestep prior to animating. 
        This could include image processing functions such as increasing contrast, unsharp masking, saturation etc. 
        The function should take AND return a three-band numpy array with shape [:, :, 3]. If your function has 
        parameters, you can pass in custom values using `partial` from `functools`: 
        `image_proc_func=partial(custom_func, param1=10)`.
        
    :param title: 
        An optional string or list of strings with a length equal to the number of timesteps in `ds`. This can be
        used to display a static title (using a string), or a dynamic title (using a list) that displays different
        text for each timestep. Defaults to False, which plots no title.
        
    :param show_date:
        An optional boolean that defines whether or not to plot date annotations for each animation frame. Defaults 
        to True, which plots date annotations based on time steps in `ds`.
        
    :param annotation_kwargs:
        An optional dict of kwargs for controlling the appearance of text annotations in the left panel to pass to the 
        matplotlib `plt.annotate` function (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html). 
        For example, `annotation_kwargs={'fontsize':20, 'color':'red', 'family':'serif'}. By default, text annotations 
        are plotted as white, size 25 mono-spaced font with a 4pt black outline in the top-right of the animation.
        
    :param onebandplot_cbar:
        An optional boolean indicating whether to include a colourbar if `ds` is a one-band array. Defaults to True.
        
    :param onebandplot_kwargs:
        An optional dict of kwargs for controlling the appearance of one-band image arrays in the left panel to pass 
        to matplotlib `plt.imshow` (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html for options).
        This only applies if an xarray with a single band is passed to `ds`. For example, a green colour scheme and
        custom stretch could be specified using: `onebandplot_kwargs={'cmap':'Greens`, 'vmin':0.2, 'vmax':0.9}`. 
        By default, one-band arrays are plotted using the 'Greys' cmap with bilinear interpolation.
        
        Two special kwargs (`tick_fontsize`, `tick_colour`) can also be passed to control the tick labels on the 
        colourbar. This can be useful for example when the tick labels are difficult to see against a dark background.
        
    :param shapefile_path:
        An optional string or list of strings giving the file paths of shapefiles to overlay on the output animation. 
        The shapefiles must be in the same projection as the input xarray dataset.
        
    :param shapefile_kwargs:
        An optional dict of kwargs to specify the appearance of the shapefile overlay to pass to `GeoSeries.plot`
        (see http://geopandas.org/reference.html#geopandas.GeoSeries.plot). For example: 
        `shapefile_kwargs = {'linewidth':2, 'edgecolor':'black', 'facecolor':"#00000000"}`
        
    :param pandasplot_kwargs:
        An optional dict of kwargs to specify the appearance of the right-hand plot to pass to `pandas.DataFrame.plot`
        (see https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.plot.html). For example: 
        `pandasplot_kwargs = {'linewidth':2, 'cmap':'viridis', 'ylim':(0, 100)}`
        
    :param time_dim:
        An optional string allowing you to override the xarray dimension used for time. Defaults to 'time'.
    
    :param x_dim:
        An optional string allowing you to override the xarray dimension used for x coordinates. Defaults to 'x'.
    
    :param y_dim:
        An optional string allowing you to override the xarray dimension used for y coordinates. Defaults to 'y'.  
        
    """    
      
    ###############
    # Setup steps #
    ############### 

    # Test if all dimensions exist in dataset
    if time_dim in ds and x_dim in ds and y_dim in ds:
        
        # Test if there is one or three bands, and that all exist in both datasets:
        if ((len(bands) == 3) | (len(bands) == 1)) & all([(b in ds.data_vars) for b in bands]):         
            
            # Import xarrays as lists of three band numpy arrays
            imagelist, vmin, vmax = _ds_to_arrraylist(ds, bands=bands, 
                                                      time_dim=time_dim, x_dim=x_dim, y_dim=y_dim, 
                                                      percentile_stretch=percentile_stretch,
                                                      image_proc_func=image_proc_func)
        
            # Get time, x and y dimensions of dataset and calculate width vs height of plot
            timesteps = len(ds[time_dim])    
            width = len(ds[x_dim])
            height = len(ds[y_dim])
            width_ratio = float(width) / float(height)
            height = 10.0 / width_ratio   

            # If title is supplied as a string, multiply out to a list with one string per timestep.
            # Otherwise, use supplied list for plot titles.
            if isinstance(title, str) or isinstance(title, bool):
                title_list = [title] * timesteps 
            else:
                title_list = title            

            # Set up annotation parameters that plt.imshow plotting for single band array images. 
            # The nested dict structure sets default values which can be overwritten/customised by the 
            # manually specified `onebandplot_kwargs`
            onebandplot_kwargs = dict({'cmap':'Greys', 'interpolation':'bilinear',
                                       'vmin': vmin, 'vmax': vmax, 'tick_colour': 'black', 'tick_fontsize': 11}, 
                                       **onebandplot_kwargs)  
            
            # Use pop to remove the two special tick kwargs from the onebandplot_kwargs dict, and save individually
            onebandplot_tick_colour = onebandplot_kwargs.pop('tick_colour')
            onebandplot_tick_fontsize = onebandplot_kwargs.pop('tick_fontsize')

            # Set up annotation parameters that control font etc. The nested dict structure sets default 
            # values which can be overwritten/customised by the manually specified `annotation_kwargs`
            annotation_kwargs = dict({'xy': (1, 1), 'xycoords':'axes fraction', 
                                      'xytext':(-5, -5), 'textcoords':'offset points', 
                                      'horizontalalignment':'right', 'verticalalignment':'top', 
                                      'fontsize':15, 'color':'white', 
                                      'path_effects': [PathEffects.withStroke(linewidth=3, foreground='black')]},
                                      **annotation_kwargs)

            # Define default plotting parameters for the overlaying shapefile(s). The nested dict structure sets 
            # default values which can be overwritten/customised by the manually specified `shapefile_kwargs`
            shapefile_kwargs = dict({'linewidth': 2, 'edgecolor': 'black', 'facecolor': "#00000000"}, 
                                     **shapefile_kwargs) 

            # Define default plotting parameters for the right-hand line plot. The nested dict structure sets 
            # default values which can be overwritten/customised by the manually specified `pandasplot_kwargs`
            pandasplot_kwargs = dict({}, **pandasplot_kwargs) 
            

            ###################
            # Initialise plot #
            ###################
            
            # Set up figure 
            fig, (ax1, ax2) = plt.subplots(ncols=2) 
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0)
            fig.set_size_inches(10.0, height * 0.5, forward=True)
            ax1.axis('off')
            ax2.margins(x=0.01)
            ax2.xaxis.label.set_visible(False)

            # Initialise axesimage objects to be updated during animation, setting extent from dims
            extents = [float(ds[x_dim].min()), float(ds[x_dim].max()), 
                       float(ds[y_dim].min()), float(ds[y_dim].max())]
            im = ax1.imshow(imagelist[0], extent=extents, **onebandplot_kwargs)

            # Initialise right panel and set y axis limits
            line_test = df.plot(ax=ax2, **pandasplot_kwargs)

            # Legend to right panel
            ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1, frameon=False) 

            # Initialise annotation objects to be updated during animation
            t = ax1.annotate('', **annotation_kwargs)

            
            #########################
            # Add optional overlays #
            #########################        
            
            # Optionally add shapefile overlay(s) from either string path or list of string paths
            if isinstance(shapefile_path, str):

                shapefile = gpd.read_file(shapefile_path)
                shapefile.plot(**shapefile_kwargs, ax=ax1)
            
            elif isinstance(shapefile_path, list):
        
                # Iterate through list of string paths
                for shapefile in shapefile_path:

                    shapefile = gpd.read_file(shapefile)
                    shapefile.plot(**shapefile_kwargs, ax=ax1) 

            # After adding shapefile, fix extents of plot
            ax1.set_xlim(extents[0], extents[1])
            ax1.set_ylim(extents[2], extents[3])

            # Optionally add colourbar for one band images
            if (len(bands) == 1) & onebandplot_cbar:                
                _add_colourbar(ax1, im, 
                               tick_fontsize=onebandplot_tick_fontsize,
                               tick_colour=onebandplot_tick_colour,
                               vmin=onebandplot_kwargs['vmin'], 
                               vmax=onebandplot_kwargs['vmax'])


            ########################################
            # Create function to update each frame #
            ########################################

            # Function to update figure
            def update_figure(frame_i):
     

                ####################
                # Plot image panel #
                ####################  

                # If possible, extract dates from time dimension
                try:

                    # Get human-readable date info (e.g. "16 May 1990")
                    ts = ds[time_dim][{time_dim:frame_i}].dt
                    year = ts.year.item()
                    month = ts.month.item()
                    day = ts.day.item()
                    date_string = '{} {} {}'.format(day, calendar.month_abbr[month], year)
                    
                except:
                    
                    date_string = ds[time_dim][{time_dim:frame_i}].values.item()

                # Create annotation string based on title and date specifications:
                title = title_list[frame_i]
                if title and show_date:
                    title_date = '{}\n{}'.format(date_string, title)
                elif title and not show_date:
                    title_date = '{}'.format(title)
                elif show_date and not title:
                    title_date = '{}'.format(date_string)           
                else:
                    title_date = ''

                # Update left panel with annotation and image
                im.set_array(imagelist[frame_i])
                t.set_text(title_date) 

                
                ########################
                # Plot linegraph panel #
                ########################              
                
                # Create list of artists to return
                artist_list = [im, t]

                # Update right panel with temporal line subset, adding each new line into artist_list
                for i, line in enumerate(line_test.lines):
                    
                    # Clip line data to current time, and get x and y values                   
                    y = df[df.index <= datetime(year=year, month=month, day=day, hour=23, minute=59)].iloc[:,i]
                    x = df[df.index <= datetime(year=year, month=month, day=day, hour=23, minute=59)].index 
                    
                    # Plot lines after stripping NaNs (this produces continuous, unbroken lines)
                    line.set_data(x[y.notnull()], y[y.notnull()])
                    artist_list.extend([line])
                    
                # Return the artists set
                return artist_list

            # Nicely space subplots
            fig.tight_layout()
            
            
            ##############################
            # Generate and run animation #
            ##############################

            # Generate animation
            ani = animation.FuncAnimation(fig=fig, func=update_figure, frames=timesteps, interval=interval, blit=True) 

            # Export as either MP4 or GIF
            if output_path[-3:] == 'mp4':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path, dpi=width_pixels / 10.0)

            elif output_path[-3:] == 'wmv':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path, dpi=width_pixels / 10.0, 
                         writer=animation.FFMpegFileWriter(fps=1000 / interval, bitrate=4000, codec='wmv2'))

            elif output_path[-3:] == 'gif':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path, dpi=width_pixels / 10.0, writer='imagemagick')

            else:
                print('    Output file type must be either .mp4, .wmv or .gif')
        
        else:        
            print('Please select either one or three bands that all exist in the input dataset')  
    
    else:
        print('At least one x, y or time dimension does not exist in the input dataset. Please use the `time_dim`,' \
              '`x_dim` or `y_dim` parameters to override the default dimension names used for plotting')


def animated_doubletimeseries(ds1, ds2, output_path, 
                              width_pixels=1000, interval=200, 
                              bands1=['red', 'green', 'blue'], bands2=['red', 'green', 'blue'],                               
                              percentile_stretch1 = (0.02, 0.98), percentile_stretch2 = (0.02, 0.98),
                              image_proc_func1=None, image_proc_func2=None,
                              title1=False, title2=False,
                              show_date1=True, show_date2=True,
                              annotation_kwargs1={}, annotation_kwargs2={},
                              onebandplot_cbar1=True, onebandplot_cbar2=True,
                              onebandplot_kwargs1={}, onebandplot_kwargs2={},
                              shapefile_path1=None, shapefile_path2=None,
                              shapefile_kwargs1={}, shapefile_kwargs2={},
                              time_dim1 = 'time', x_dim1 = 'x', y_dim1 = 'y',
                              time_dim2 = 'time', x_dim2 = 'x', y_dim2 = 'y'):
    
    """
    Takes two xarray time series and animates both side-by-side as either three-band (e.g. true or false colour) 
    or single-band animations, allowing changes in the landscape to be compared across time.
    
    Animations can be exported as .mp4 (ideal for Twitter/social media), .wmv (ideal for Powerpoint) and .gif 
    (ideal for all purposes, but can have large file sizes) format files, and customised to include titles and 
    date annotations for each panel or use different input bands from each dataset. For example, true and false 
    colour band combinations could be plotted at the same time, or different products (i.e. NBAR and NBART) or 
    cloud masking algorithms could be compared. 
    A shapefile boundary can be added to the output animation by providing a path to the shapefile.
    
    This function can be used to produce visually appealing cloud-free animations when used in combination with 
    the `load_clearlandsat` function from `dea-notebooks/10_Scripts/DEADataHandling`.
    
    Last modified: October 2018
    Author: Robbi Bishop-Taylor, Sean Chua, Bex Dunn
    
    :param ds1: 
        An xarray dataset with multiple time steps (i.e. multiple observations along the `time_dim` dimension) to be 
        plotted in the left panel of the animation.
        
    :param ds2: 
        A matching xarray dataset with the same number of pixels as ds1, to be plotted in the right panel of the
        animation. ds1 and ds2 do not need to have exactly the same number of timesteps, but the animation will 
        only continue up until the length of the shorted dataset (i.e. if `ds1` has 10 timesteps and `ds2` has 5, 
        the animation will continue for 5 timesteps).
        
    :param output_path: 
        A string giving the output location and filename of the resulting animation. File extensions of '.mp4', 
        '.wmv' and '.gif' are accepted.
        
    :param width_pixels:
        An optional integer defining the output width in pixels for the resulting animation. The height of the 
        animation is set automatically based on the dimensions/ratio of `ds1`. Defaults to 
        1000 pixels wide.
        
    :param interval:
        An optional integer defining the milliseconds between each animation frame used to control the speed of 
        the output animation. Higher values result in a slower animation. Defaults to 200 milliseconds between 
        each frame.
        
    :param bands1:
        An optional list of either one or three bands to be plotted, all of which must exist in `ds1`.
        Defaults to `['red', 'green', 'blue']`.
    
    :param bands2:
        An optional list of either one or three bands to be plotted, all of which must exist in `ds2`.
        Defaults to `['red', 'green', 'blue']`.        
      
    :param percentile_stretch1:        
        An optional tuple of two floats that can be used to clip one or three-band arrays in the left `ds1` panel
        by percentiles to produce a more vibrant, visually attractive image that is not affected by outliers/extreme 
        values. The default is `(0.02, 0.98)` which is equivalent to xarray's `robust=True`.
    
    :param percentile_stretch2:
        An optional tuple of two floats that can be used to clip one or three-band arrays in the right `ds2` panel
        by percentiles to produce a more vibrant, visually attractive image that is not affected by outliers/extreme 
        values. The default is `(0.02, 0.98)` which is equivalent to xarray's `robust=True`.
        
    :param image_proc_func1:
        An optional function can be passed to modify three-band arrays in the left `ds1` panel for each timestep 
        prior to animating. This could include image processing functions such as increasing contrast, unsharp 
        masking, saturation etc. The function should take AND return a three-band numpy array with shape [:, :, 3]. 
        If your function has parameters, you can pass in custom values using `partial` from `functools`: 
        `image_proc_func=partial(custom_func, param1=10)`.
        
    :param image_proc_func2:
        An optional function can be passed to modify three-band arrays in the right `ds1` panel for each timestep 
        prior to animating. This could include image processing functions such as increasing contrast, unsharp 
        masking, saturation etc. The function should take AND return a three-band numpy array with shape [:, :, 3]. 
        If your function has parameters, you can pass in custom values using `partial` from `functools`: 
        `image_proc_func=partial(custom_func, param1=10)`.
    :param title1: 
        An optional string or list of strings with a length equal to the number of timesteps in `ds1`. This can be
        used to display a static title for the left panel (using a string), or a dynamic title (using a list)
        that displays different text for each timestep. Defaults to False, which plots no title.
        
    :param title2: 
        An optional string or list of strings with a length equal to the number of timesteps in `ds2`. This can be
        used to display a static title for the left panel (using a string), or a dynamic title (using a list)
        that displays different text for each timestep. Defaults to False, which plots no title.
        
    :param show_date1:
        An optional boolean that defines whether or not to plot date annotations for each animation frame in the 
        left panel. Defaults to True, which plots date annotations for `ds1`.
    
    :param show_date2:
        An optional boolean that defines whether or not to plot date annotations for each animation frame in the 
        right panel. Defaults to True, which plots date annotations for `ds2`.
    :param annotation_kwargs1:
        An optional dict of kwargs for controlling the appearance of `ds1` text annotations to pass to 
        matplotlib `plt.annotate`  (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html). 
        For example, `annotation_kwargs1={'fontsize':20, 'color':'red', 'family':'serif'}. By default, text 
        annotations are white, size 15 mono-spaced font with a 3pt black outline in the panel's top-right. 
    
    :param annotation_kwargs2:
        An optional dict of kwargs for controlling the appearance of the `ds2` text annotations to pass 
        to matplotlib `plt.annotate` (see above).
        
    :param onebandplot_cbar1:
        An optional boolean indicating whether to include a colourbar if `ds1` is a one-band array. Defaults to True.
        
    :param onebandplot_cbar2:
        An optional boolean indicating whether to include a colourbar if `ds2` is a one-band array. Defaults to True.
        
    :param onebandplot_kwargs1:
        An optional dict of kwargs for controlling the appearance of `ds1` one-band image arrays to pass to 
        matplotlib `plt.imshow` (see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html for options).
        This only applies if an xarray with a single band is passed to d1. For example, a green colour scheme and
        custom stretch can be specified using: `onebandplot_kwargs1={'cmap':'Greens`, 'vmin':0.2, 'vmax':0.9}`. 
        By default, one-band arrays are plotted using the 'Greys' cmap with bilinear interpolation.
        
        Two special kwargs (`tick_fontsize`, `tick_colour`) can also be passed to control the tick labels on the 
        colourbar. This can be useful for example when the tick labels are difficult to see against a dark background.
    
    :param onebandplot_kwargs2:
        An optional dict of kwargs for controlling the appearance of `ds2` one-band image arrays to 
        pass to matplotlib `plt.imshow`; only applies if an xarray with a single band is passed to `d2` (see above).
    :param shapefile_path1:
        An optional string or list of strings giving the file paths of shapefiles to overlay on the left `ds1` panel. 
        The shapefiles must be in the same projection as the input xarray dataset.
    :param shapefile_path2:
        An optional string or list of strings giving the file paths of shapefiles to overlay on the right `ds2` panel. 
        The shapefiles must be in the same projection as the input xarray dataset.
        
    :param shapefile_kwargs1:
        An optional dict of kwargs to specify the appearance of the left `ds1` panel shapefile overlay to pass to 
        `GeoSeries.plot` (see http://geopandas.org/reference.html#geopandas.GeoSeries.plot). For example: 
         `shapefile_kwargs = {'linewidth':2, 'edgecolor':'black', 'facecolor':"#00000000"}`
    :param shapefile_kwargs2:
        An optional dict of kwargs to specify the appearance of the right `ds2` panelshapefile overlay. For example: 
        `shapefile_kwargs = {'linewidth':2, 'edgecolor':'black', 'facecolor':"#00000000"}`
        
    :param time_dim1:
        An optional string allowing you to override the xarray dimension used for time in `ds1`. 
        Defaults to 'time'.
    
    :param time_dim2:
        An optional string allowing you to override the xarray dimension used for time in `ds2`. 
        Defaults to 'time'.
    
    :param x_dim1:
        An optional string allowing you to override the xarray dimension used for x coordinates in `ds1`. 
        Defaults to 'x'.
        
    :param x_dim2:
        An optional string allowing you to override the xarray dimension used for x coordinates in `ds2`. 
        Defaults to 'x'.
    
    :param y_dim1:
        An optional string allowing you to override the xarray dimension used for y coordinates in `ds1`. 
        Defaults to 'y'
    
    :param y_dim2:
        An optional string allowing you to override the xarray dimension used for y coordinates in `ds2`. 
        Defaults to 'y'
        
    """
    
    ###############
    # Setup steps #
    ###############
    
    # Test if all dimensions exist in dataset
    if time_dim1 in ds1 and x_dim1 in ds1 and y_dim1 in ds1 and time_dim2 in ds2 and x_dim2 in ds2 and y_dim2 in ds2:
        
        # First test if there are three bands, and that all exist in both datasets:
        if ((len(bands1) == 3) | (len(bands1) == 1)) & all([(b1 in ds1.data_vars) for b1 in bands1]) & \
           ((len(bands2) == 3) | (len(bands2) == 1)) & all([(b2 in ds2.data_vars) for b2 in bands2]):  
    
            # Import xarrays as lists of three band numpy arrays
            imagelist1, vmin1, vmax1 = _ds_to_arrraylist(ds1, bands=bands1, 
                                                         time_dim=time_dim1, x_dim=x_dim1, y_dim=y_dim1, 
                                                         percentile_stretch=percentile_stretch1,
                                                         image_proc_func=image_proc_func1)
            imagelist2, vmin2, vmax2 = _ds_to_arrraylist(ds2, bands=bands2, 
                                                         time_dim=time_dim2, x_dim=x_dim2, y_dim=y_dim2, 
                                                         percentile_stretch=percentile_stretch2,
                                                         image_proc_func=image_proc_func2)
    
            # Get time, x and y dimensions of dataset 
            timesteps1 = len(ds1[time_dim1])  
            timesteps2 = len(ds2[time_dim2])
            width1 = len(ds1[x_dim1])
            width2 = len(ds2[x_dim2])
            height1 = len(ds1[y_dim1])
            height2 = len(ds2[y_dim2])

            # Calculate width vs height of plot
            width_ratio = float(width1) / float(height1)
            width = width1
            height = 10.0 / width_ratio

            # If title is supplied as a string, multiply out to a list with one string per timestep.
            # Otherwise, use supplied list for plot titles.
            if isinstance(title1, str) or isinstance(title1, bool):
                title_list1 = [title1] * timesteps1   
            else:
                title_list1 = title1

            # If title is supplied as a string, multiply out to a list with one string per timestep
            if isinstance(title2, str) or isinstance(title2, bool):
                title_list2 = [title2] * timesteps2  
            else:
                title_list2 = title2       

            # Set up annotation parameters that plt.imshow plotting for single band array images. 
            # The nested dict structure sets default values which can be overwritten/customised by the 
            # manually specified `onebandplot_kwargs`
            onebandplot_kwargs1 = dict({'cmap':'Greys', 'interpolation':'bilinear',                                  
                                        'vmin':vmin1, 'vmax':vmax1, 'tick_colour': 'black', 'tick_fontsize': 11},
                                        **onebandplot_kwargs1) 
            
            # Use pop to remove the two special tick kwargs from the onebandplot_kwargs dict, and save individually
            onebandplot_tick_colour1 = onebandplot_kwargs1.pop('tick_colour')
            onebandplot_tick_fontsize1 = onebandplot_kwargs1.pop('tick_fontsize')

            onebandplot_kwargs2 = dict({'cmap':'Greys', 'interpolation':'bilinear',
                                        'vmin':vmin2, 'vmax':vmax2, 'tick_colour': 'black', 'tick_fontsize': 11},
                                        **onebandplot_kwargs2) 
            
            # Use pop to remove the two special tick kwargs from the onebandplot_kwargs dict, and save individually
            onebandplot_tick_colour2 = onebandplot_kwargs2.pop('tick_colour')
            onebandplot_tick_fontsize2 = onebandplot_kwargs2.pop('tick_fontsize')

            # Set up annotation parameters that control font etc. The nested dict structure sets default 
            # values which can be overwritten/customised by the manually specified `annotation_kwargs`
            annotation_kwargs1 = dict({'xy': (1, 1), 'xycoords':'axes fraction', 
                                       'xytext':(-5, -5), 'textcoords':'offset points', 
                                       'horizontalalignment':'right', 'verticalalignment':'top', 
                                       'fontsize':15, 'color':'white', 
                                       'path_effects':[PathEffects.withStroke(linewidth=3, foreground='black')]},
                                       **annotation_kwargs1)

            annotation_kwargs2 = dict({'xy': (1, 1), 'xycoords':'axes fraction', 
                                       'xytext':(-5, -5), 'textcoords':'offset points', 
                                       'horizontalalignment':'right', 'verticalalignment':'top', 
                                       'fontsize':15, 'color':'white', 
                                       'path_effects':[PathEffects.withStroke(linewidth=3, foreground='black')]},
                                       **annotation_kwargs2)

            # Define default plotting parameters for the overlaying shapefile(s). The nested dict structure sets 
            # default values which can be overwritten/customised by the manually specified `shapefile_kwargs`
            shapefile_kwargs1 = dict({'linewidth': 2, 'edgecolor': 'black', 'facecolor': "#00000000"}, 
                                     **shapefile_kwargs1) 

            shapefile_kwargs2 = dict({'linewidth': 2, 'edgecolor': 'black', 'facecolor': "#00000000"}, 
                                     **shapefile_kwargs2) 
            

            ###################
            # Initialise plot #
            ################### 
        
            # Set up figure
            fig, (ax1, ax2) = plt.subplots(ncols=2) 
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            fig.set_size_inches(10.0, height * 0.5, forward=True)
            ax1.axis('off')
            ax2.axis('off')

            # Initialise axesimage objects to be updated during animation
            extents1 = [float(ds1[x_dim1].min()), float(ds1[x_dim1].max()), 
                        float(ds1[y_dim1].min()), float(ds1[y_dim1].max())]
            extents2 = [float(ds2[x_dim2].min()), float(ds2[x_dim2].max()), 
                        float(ds2[y_dim2].min()), float(ds2[y_dim2].max())]
            im1 = ax1.imshow(imagelist1[0], extent=extents1, **onebandplot_kwargs1)
            im2 = ax2.imshow(imagelist2[0], extent=extents2, **onebandplot_kwargs2)

            # Initialise annotation objects to be updated during animation
            t1 = ax1.annotate('', **annotation_kwargs1)   
            t2 = ax2.annotate('', **annotation_kwargs2)  


            #########################
            # Add optional overlays #
            #########################        

            # Optionally add shapefile overlay(s) from either string path or list of string paths
            if isinstance(shapefile_path1, str):

                shapefile = gpd.read_file(shapefile_path1)
                shapefile.plot(**shapefile_kwargs1, ax=ax1)

            elif isinstance(shapefile_path1, list):

                # Iterate through list of string paths
                for shapefile in shapefile_path1:

                    shapefile = gpd.read_file(shapefile)
                    shapefile.plot(**shapefile_kwargs1, ax=ax1) 

            # Optionally add shapefile overlay(s) from either string path or list of string paths
            if isinstance(shapefile_path2, str):

                shapefile = gpd.read_file(shapefile_path2)
                shapefile.plot(**shapefile_kwargs2, ax=ax2)

            elif isinstance(shapefile_path2, list):

                # Iterate through list of string paths
                for shapefile in shapefile_path2:

                    shapefile = gpd.read_file(shapefile)
                    shapefile.plot(**shapefile_kwargs2, ax=ax2) 

            # After adding shapefile, fix extents of plot
            ax1.set_xlim(extents1[0], extents1[1])
            ax1.set_ylim(extents1[2], extents1[3])
            ax2.set_xlim(extents2[0], extents2[1])
            ax2.set_ylim(extents2[2], extents2[3])

            # Optionally add colourbars for one band images
            if (len(bands1) == 1) & onebandplot_cbar1:                
                _add_colourbar(ax1, im1, 
                               tick_fontsize=onebandplot_tick_fontsize1,
                               tick_colour=onebandplot_tick_colour1,
                               vmin=onebandplot_kwargs1['vmin'], 
                               vmax=onebandplot_kwargs1['vmax'])

            if (len(bands2) == 1) & onebandplot_cbar2:                
                _add_colourbar(ax2, im2, 
                               tick_fontsize=onebandplot_tick_fontsize2,
                               tick_colour=onebandplot_tick_colour2,
                               vmin=onebandplot_kwargs2['vmin'], 
                               vmax=onebandplot_kwargs2['vmax'])

            # Function to update figure
            def update_figure(frame_i):

                ####################
                # Plot first panel #
                ####################  

                # If possible, extract dates from time dimension
                try:

                    # Get human-readable date info (e.g. "16 May 1990")
                    ts = ds1[time_dim1][{time_dim1:frame_i}].dt
                    year = ts.year.item()
                    month = ts.month.item()
                    day = ts.day.item()
                    date_string = '{} {} {}'.format(day, calendar.month_abbr[month], year)

                except:

                    date_string = ds1[time_dim1][{time_dim1:frame_i}].values.item()

                # Create annotation string based on title and date specifications:
                title1 = title_list1[frame_i]
                if title1 and show_date1:
                    title_date1 = '{}\n{}'.format(date_string, title1)
                elif title1 and not show_date1:
                    title_date1 = '{}'.format(title1)
                elif show_date1 and not title1:
                    title_date1 = '{}'.format(date_string)           
                else:
                    title_date1 = ''

                # Update figure for frame
                im1.set_array(imagelist1[frame_i])
                t1.set_text(title_date1) 


                #####################
                # Plot second panel #
                ##################### 

                # If possible, extract dates from time dimension
                try:

                    # Get human-readable date info (e.g. "16 May 1990")
                    ts = ds2[time_dim2][{time_dim2:frame_i}].dt
                    year = ts.year.item()
                    month = ts.month.item()
                    day = ts.day.item()
                    date_string = '{} {} {}'.format(day, calendar.month_abbr[month], year)

                except:

                    date_string = ds2[time_dim2][{time_dim2:frame_i}].values.item()

                # Create annotation string based on title and date specifications:
                title2 = title_list2[frame_i]
                if title2 and show_date2:
                    title_date2 = '{}\n{}'.format(date_string, title2)
                elif title2 and not show_date2:
                    title_date2 = '{}'.format(title2)
                elif show_date2 and not title2:
                    title_date2 = '{}'.format(date_string)         
                else:
                    title_date2 = ''

                # Update figure for frame
                im2.set_array(imagelist2[frame_i])
                t2.set_text(title_date2) 

                # Return the artists set
                return [im1, im2, t1, t2]


            ##############################
            # Generate and run animation #
            ##############################

            # Generate animation
            frames_to_run = min(timesteps1, timesteps2)
            print('Generating {} frame animation (i.e. timesteps in shortest dataset)'.format(frames_to_run))
            ani = animation.FuncAnimation(fig, update_figure, frames=frames_to_run, interval=interval, blit=True)

            # Export as either MP4 or GIF
            if output_path[-3:] == 'mp4':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path, dpi=width_pixels / 10.0)

            elif output_path[-3:] == 'wmv':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path, dpi=width_pixels / 10.0, 
                         writer=animation.FFMpegFileWriter(fps=1000 / interval, bitrate=4000, codec='wmv2'))

            elif output_path[-3:] == 'gif':
                print('    Exporting animation to {}'.format(output_path))
                ani.save(output_path, dpi=width_pixels / 10.0, writer='imagemagick')

            else:
                print('    Output file type must be either .mp4, .wmv or .gif')

        else:        
            print('Please select either one or three bands that all exist in the input datasets')  
            
    else:
        print('At least one x, y or time dimension does not exist in `ds1` or `ds2`. Please use the `time_dim`,' \
              '`x_dim` or `y_dim` parameters to override the default dimension names used for plotting') 

        
# Define function to convert xarray dataset to list of one or three band numpy arrays
def _ds_to_arrraylist(ds, bands, time_dim, x_dim, y_dim, percentile_stretch, image_proc_func=None): 

    """
    Converts an xarray dataset to a list of numpy arrays for plt.imshow plotting
    """

    # Compute percents
    p_low, p_high = ds[bands].to_array().quantile(percentile_stretch).values

    array_list = []
    for i, timestep in enumerate(ds[time_dim]):

        # Select single timestep from the data array
        ds_i = ds[{time_dim: i}]

        # Get shape of array
        x = len(ds[x_dim])
        y = len(ds[y_dim])

        if len(bands) == 1:    

            # Create new one band array
            img_toshow = exposure.rescale_intensity(ds_i[bands[0]].values, 
                                                    in_range=(p_low, p_high),
                                                    out_range='image')

        else:

            # Create new three band array                
            rawimg = np.zeros((y, x, 3), dtype=np.float32)

            # Add xarray bands into three dimensional numpy array
            for band, colour in enumerate(bands):

                rawimg[:, :, band] = ds_i[colour].values

            # Stretch contrast using percentile values
            img_toshow = exposure.rescale_intensity(rawimg, 
                                                    in_range=(p_low, p_high),
                                                    out_range=(0, 1.0))

            # Optionally image processing
            if image_proc_func:
                
                img_toshow = image_proc_func(img_toshow).clip(0, 1)

        array_list.append(img_toshow)

    return array_list, p_low, p_high


def _add_colourbar(ax, im, vmin, vmax, cmap='Greys', tick_fontsize=15, tick_colour='black'):

    """
    Add a nicely formatted colourbar to an animation panel
    """

    # Add colourbar
    axins2 = inset_axes(ax, width='97%', height='4%', loc=8, borderpad=1) 
    plt.gcf().colorbar(im, cax=axins2, orientation='horizontal', ticks=np.linspace(vmin, vmax, 3)) 
    axins2.xaxis.set_ticks_position('top')
    axins2.tick_params(axis='x', colors=tick_colour, labelsize=tick_fontsize) 
    
    # Justify left and right labels to edge of plot
    axins2.get_xticklabels()[0].set_horizontalalignment('left')
    axins2.get_xticklabels()[-1].set_horizontalalignment('right')
    labels = [item.get_text() for item in axins2.get_xticklabels()]
    labels[0] = '  ' + labels[0]
    labels[-1] = labels[-1] + '  '
    
        
def _degree_to_zoom_level(l1, l2, margin=0.0):
    
    """
    Helper function to set zoom level for `display_map`
    """
    
    degree = abs(l1 - l2) * (1 + margin)
    zoom_level_int = 0
    if degree != 0:
        zoom_level_float = math.log(360 / degree) / math.log(2)
        zoom_level_int = int(zoom_level_float)
    else:
        zoom_level_int = 18
    return zoom_level_int
