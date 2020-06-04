# deafrica_wetlandstools.py

'''
Description: This file contains a set of python functions for working with
the Wetlands Insight Tool (WIT)

License: The code in this notebook is licensed under the Apache License,
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Digital Earth 
Africa data is licensed under the Creative Commons by Attribution 4.0 
license (https://creativecommons.org/licenses/by/4.0/).

Contact: If you need assistance, please post a question on the Open Data 
Cube Slack channel (http://slack.opendatacube.org/) or on the GIS Stack 
Exchange (https://gis.stackexchange.com/questions/ask?tags=open-data-cube) 
using the `open-data-cube` tag (you can view previously asked questions 
here: https://gis.stackexchange.com/questions/tagged/open-data-cube). 

If you would like to report an issue with this script, you can file one on 
Github: https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues/new

Functions included:
    WIT_drill
    thresholded_tasseled_cap
    animated_timeseries_WIT
    _ds_to_arrraylist
    _add_colourbar
    

Last modified: Feb 2020

'''


# Import required packages

import deafrica_plotting
import deafrica_datahandling
from deafrica_datahandling import wofs_fuser
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

import datacube

# import datetime
import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.mask
import rasterio.features
import dask

import seaborn as sns
import sys
import xarray as xr
from multiprocessing import Pool
from datacube.storage.masking import make_mask
from datacube.utils import masking
from datacube.utils import geometry
import datacube.utils.rio


def WIT_drill(gdf_poly,
              time,
              min_gooddata=0.80,
              TCW_threshold=-6000,
              export_csv=None,
              dask_chunks=None):
    """
    The Wetlands Insight Tool. This function loads FC, WOfS, Landsat-ARD,
    and calculate tasseled cap wetness, in order to determine the dominant
    land cover class within a polygon at each satellite observation.

    The output is a pandas dataframe containing a timeseries of the relative
    fractions of each class at each time-step. This forms the input to produce
    a stacked line-plot.

    Last modified: Feb 2020

    Parameters
    ----------  
    gdf_poly : geopandas.GeoDataFrame
        The dataframe must only contain a single row,
        containing the polygon you wish to interrograte.
    time : tuple
        a tuple containing the time range over which to run the WIT.
        e.g. ('2015-01' , '2019-12')
    min_gooddata : Float, optional
        A number between 0 and 1 (e.g 0.8) indicating the minimum percentage
        of good quality pixels required for a satellite observation to be loaded
        and therefore included in the WIT plot.  Defaults to 0.8, which should
        be considered a minimum percentage.
    TCW_threshold : Int, optional
        The tasseled cap wetness threshold, beyond which a pixel will be 
        considered 'wet'. Defaults to -6000. Consider the surface reflectance
        scaling of the Landsat product when adjusting this (C2 = 1-65,535) 
    export_csv : str, optional
        To export the returned pandas dataframe provide
        a location string (e.g. 'output/results.csv')
    dask_chunks : dict, optional
        To lazily load the datasets using dask, pass a dictionary containing
        the dimensions over which to chunk e.g. {'time':-1, 'x':250, 'y':250}.
        The function is not currently set up to handle dask arrays very well, so
        memory efficieny using dask will be of limited use here.
        
    Returns
    -------
    PolyDrill_df : Pandas.Dataframe
        A pandas dataframe containing the timeseries of relative fractions
        of each land cover class (WOfs, FC, TCW) 

    """

    print("working on polygon: " +str(gdf_poly.drop('geometry', axis=1).values) + ".  ")

    # make quaery from polygon
    geom = geometry.Geometry(
        gdf_poly.geometry.values[0].__geo_interface__, geometry.CRS(
            "epsg:4326")
    )
    query = {"geopolygon": geom, "time": time}

    # set Sandbox configs to load COG's faster
    datacube.utils.rio.set_default_rio_config(aws="auto", cloud_defaults=True)
    
    # Create a datacube instance
    dc = datacube.Datacube(app="wetlands insight tool")

    # find UTM crs for location
    crs = deafrica_datahandling.mostcommon_crs(
        dc=dc, product="usgs_ls8c_level2_2", query=query
    )

    # load landsat 5,7,8 data
    ls578_ds = deafrica_datahandling.load_ard(
        dc=dc,
        products=["usgs_ls8c_level2_2"],
        output_crs=crs,
        min_gooddata=min_gooddata,
        measurements=["red", "green", "blue", "nir", "swir_1", "swir_2"],
        align=(15, 15),
        dask_chunks=dask_chunks,
        group_by='solar_day',
        resolution=(-30, 30),
        **query,
    )

    # mask the data with our original polygon to remove extra data
    data = ls578_ds
    mask = rasterio.features.geometry_mask(
        [geom.to_crs(data.geobox.crs) for geoms in [geom]],
        out_shape=data.geobox.shape,
        transform=data.geobox.affine,
        all_touched=False,
        invert=False,
    )

    # mask the data with the polygon
    mask_xr = xr.DataArray(mask, dims=("y", "x"))
    ls578_ds = data.where(mask_xr == False)
    print("size of wetlands array: " +
          str(ls578_ds.isel(time=1).red.values.shape))

    ls578_ds = ls578_ds.compute()

    # calculate tasselled cap wetness within masked AOI
    print("calculating tasseled cap index ")
    tci = thresholded_tasseled_cap(
        ls578_ds, wetness_threshold=TCW_threshold, drop=True, drop_tc_bands=True
    )
    # select only finite values (over threshold values)
    tcw = xr.ufuncs.isfinite(tci.wetness_thresholded)
    # #reapply the polygon mask
    tcw = tcw.where(mask_xr == False)

    print("Loading WOfS layers ")
    wofls = dc.load(
        product="ga_ls8c_wofs_2",
        like=ls578_ds,
        fuse_func=wofs_fuser,
        dask_chunks=dask_chunks,
    )
    wofls = wofls.where(wofls.time == tcw.time)
    # #reapply the polygon mask
    wofls = wofls.where(mask_xr == False)
    wofls = wofls.compute()

    wet_wofs = wofls.where(wofls.water == 128)

    # use bit values for wet (128) and terrain/low-angle (8)
    shadow_wofs = wofls.where(wofls.water == 136)
    # bit values for wet (128) and sea (4)
    sea_wofs = wofls.where(wofls.water == 132)
    # bit values for wet (128) and sea (4) and terrain/low-angle (8)
    sea_shadow_wofs = wofls.where(wofls.water == 140)

    # load Fractional cover
    print("Loading fractional Cover")
    # load fractional cover
    fc_ds = dc.load(
        product="ga_ls8c_fractional_cover_2",
        dask_chunks=dask_chunks,
        like=ls578_ds,
        measurements=["pv", "npv", "bs"],
    )
    # use landsat data set to cloud mask FC
    fc_ds = fc_ds.where(ls578_ds.red)

    # mask with polygon
    fc_ds = fc_ds.where(mask_xr == False)
    fc_ds = fc_ds.compute()

    fc_ds_noTCW = fc_ds.where(tcw == False)

    print("Generating classification")
    # match timesteps
    fc_ds_noTCW = fc_ds_noTCW.where(fc_ds_noTCW.time == tcw.time)

    # following robbi's advice, cast the dataset to a dataarray
    maxFC = fc_ds_noTCW.to_array(dim="variable", name="maxFC")

    # turn FC array into integer only as nanargmax doesn't seem to handle floats the way we want it to
    FC_int = maxFC.astype("int8")

    # use numpy.nanargmax to get the index of the maximum value along the variable dimension
    # BSPVNPV=np.nanargmax(FC_int, axis=0)
    BSPVNPV = FC_int.argmax(dim="variable")

    FC_mask = xr.ufuncs.isfinite(maxFC).all(dim="variable")

    # #re-mask with nans to remove no-data
    BSPVNPV = BSPVNPV.where(FC_mask)
    # restack the Fractional cover dataset all together
    # CAUTION:ARGMAX DEPENDS ON ORDER OF VARIABALES IN
    # DATASET, THESE WILL BE DIFFERENT FOR DIFFERENT COLLECTIONS.
    # NEED TO ADJUST 0,1,2 BELOW DEPENDING ON ORDER OF FC VARIABLES
    # IN THE DATASET.
    FC_dominant = xr.Dataset(
        {
            "BS": (BSPVNPV == 2).where(FC_mask),
            "PV": (BSPVNPV == 0).where(FC_mask),
            "NPV": (BSPVNPV == 1).where(FC_mask),
        }
    )
    # count number of Fractional Cover pixels for each cover type in area of interest
    FC_count = FC_dominant.sum(dim=["x", "y"])

    # number of pixels in area of interest
    pixels = (mask_xr == 0).sum(dim=["x", "y"])

    # count number of tcw pixels
    tcw_pixel_count = tcw.sum(dim=["x", "y"])

    #     return FC_dominant, FC_mask, BSPVNPV, fc_ds, ls578_ds
    # number of pixels in area of interest
    pixels = (mask_xr == 0).sum(dim=["x", "y"])

    wofs_pixels = (
        wet_wofs.water.count(dim=["x", "y"])
        + shadow_wofs.water.count(dim=["x", "y"])
        + sea_wofs.water.count(dim=["x", "y"])
        + sea_shadow_wofs.water.count(dim=["x", "y"])
    )

    # count percentage of area of wofs
    wofs_area_percent = (wofs_pixels / pixels) * 100

    # count number of tcw pixels
    tcw_pixel_count = tcw.sum(dim=["x", "y"])

    # calculate percentage area wet
    tcw_area_percent = (tcw_pixel_count / pixels) * 100

    # calculate wet not wofs
    tcw_less_wofs = tcw_area_percent - wofs_area_percent

    # Fractional cover pixel count method
    # Get number of FC pixels, divide by total number of pixels per polygon
    # Work out the number of nodata pixels in the data, so that we can graph the variables by number of observed pixels.
    Bare_soil_percent = (FC_count.BS / pixels) * 100
    Photosynthetic_veg_percent = (FC_count.PV / pixels) * 100
    NonPhotosynthetic_veg_percent = (FC_count.NPV / pixels) * 100
    NoData = (
        100
        - wofs_area_percent
        - tcw_less_wofs
        - Photosynthetic_veg_percent
        - NonPhotosynthetic_veg_percent
        - Bare_soil_percent
    )
    NoDataPixels = (NoData / 100) * pixels

    # Fractional cover pixel count method
    # Get number of FC pixels, divide by total number of pixels per polygon
    Bare_soil_percent2 = (FC_count.BS / (pixels - NoDataPixels)) * 100
    Photosynthetic_veg_percent2 = (FC_count.PV / (pixels - NoDataPixels)) * 100
    NonPhotosynthetic_veg_percent2 = (
        FC_count.NPV / (pixels - NoDataPixels)) * 100

    # count percentage of area of wofs
    wofs_area_percent2 = (wofs_pixels / (pixels - NoDataPixels)) * 100
    # wofs_area_percent
    wofs_area_percent = (wofs_pixels / pixels) * 100
    # count number of tcw pixels
    tcw_pixel_count2 = tcw.sum(dim=["x", "y"])

    # calculate percentage area wet
    tcw_area_percent2 = (tcw_pixel_count2 / (pixels - NoDataPixels)) * 100

    # calculate wet not wofs
    tcw_less_wofs2 = tcw_area_percent2 - wofs_area_percent2

    # last check for timestep matching before we plot
    wofs_area_percent2 = wofs_area_percent2.where(
        wofs_area_percent2.time == Bare_soil_percent2.time
    )
    Bare_soil_percent2 = Bare_soil_percent2.where(
        Bare_soil_percent2.time == wofs_area_percent2.time
    )
    Photosynthetic_veg_percent2 = Photosynthetic_veg_percent2.where(
        Photosynthetic_veg_percent2.time == wofs_area_percent2.time
    )
    NonPhotosynthetic_veg_percent2 = NonPhotosynthetic_veg_percent2.where(
        NonPhotosynthetic_veg_percent2.time == wofs_area_percent2.time
    )

    # start setup of dataframe by adding only one dataset
    WOFS_df = pd.DataFrame(
        data=wofs_area_percent2.data,
        index=wofs_area_percent2.time.values,
        columns=["wofs_area_percent"],
    )

    # add data into pandas dataframe for export
    WOFS_df["wet_percent"] = tcw_less_wofs2.data
    WOFS_df["green_veg_percent"] = Photosynthetic_veg_percent2.data
    WOFS_df["dry_veg_percent"] = NonPhotosynthetic_veg_percent2.data
    WOFS_df["bare_soil_percent"] = Bare_soil_percent2.data

    # call the composite dataframe something sensible, like PolyDrill
    PolyDrill_df = WOFS_df.round(2)

    # save the csv of the output data used to create the stacked plot for the polygon drill
    if export_csv:
        print('exporting csv: ' + export_csv)
        PolyDrill_df.to_csv(
            export_csv, index_label="Datetime"
        )

    ls578_ds = None
    data = None
    fc_ds = None
    wofls = None
    tci = None

    return PolyDrill_df



def thresholded_tasseled_cap(sensor_data, tc_bands=['greenness', 'brightness', 'wetness'],
                             greenness_threshold=700, brightness_threshold=4000,
                             wetness_threshold=-6000, drop=True, drop_tc_bands=True):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    NOTE: We need to think more about the thresholds to account for the different
    scalings on the C2 Landsat product. Only the wetness threshold has been looked at,
    the other thresholds are imported from DEA and are almost certainly incorrect.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    Computes thresholded tasseled cap wetness, greenness and brightness bands 
    from a six band xarray dataset, and returns a new xarray dataset with old bands
    optionally dropped.
    
    Parameters
    ----------
    sensor_data: xarray.Dataset
        Input xarray dataset with six optical Landsat bands
    tc_bands: list, optional
        Tasseled cap bands to compute e.g. ['wetness', 'greenness','brightness']
    greeness_threshold: Int, optional
        The tasseled cap greeness threshold, beyond which a pixel will be 
        considered 'green'. Defaults to 700. Consider the surface reflectance
        scaling of the Landsat product when adjusting this (C2 = 1-65,535) 
    brightness_threshold: Int, optional
        The tasseled cap brightness threshold, beyond which a pixel will be 
        considered 'bright'. Defaults to 4000. Consider the surface reflectance
        scaling of the Landsat product when adjusting this (C2 = 1-65,535)
    wetness_threshold: Int, optional
        The tasseled cap wetness threshold, beyond which a pixel will be 
        considered 'wet'. Defaults to -6000. Consider the surface reflectance
        scaling of the Landsat product when adjusting this (C2 = 1-65,535)
    drop: boolean, optional
        if 'drop = False', return all original Landsat bands
    drop_tc_bands: boolean, optional 
        if 'drop_tc_bands = False', return all unthresholded tasseled 
        cap bands as well as the thresholded bands
    
    Returns
    -------
    output_array : xarray.dataset
        Dataset containing computed thresholded tasseled cap bands
    
    
    Last modified: Feb 2020
    """

    # Copy input dataset
    output_array = sensor_data.copy(deep=True)

    # Coefficients for each tasseled cap band
    wetness_coeff = {'blue': 0.0315, 'green': 0.2021, 'red': 0.3102,
                     'nir': 0.1594, 'swir_1': -0.6806, 'swir_2': -0.6109}

    greenness_coeff = {'blue': -0.1603, 'green': -0.2819, 'red': -0.4934,
                       'nir': 0.7940, 'swir_1': -0.0002, 'swir_2': -0.1446}

    brightness_coeff = {'blue': 0.2043, 'green': 0.4158, 'red': 0.5524,
                        'nir': 0.5741, 'swir_1': 0.3124, 'swir_2': 0.2303}

    # Dict to use correct coefficients for each tasseled cap band
    analysis_coefficient = {'wetness': wetness_coeff,
                            'greenness': greenness_coeff,
                            'brightness': brightness_coeff}

    # make dictionary of thresholds for wetness, brightness and greenness
    # FIXME:add statistical and/or secant thresholding options?

    analysis_thresholds = {'wetness_threshold': wetness_threshold,
                           'greenness_threshold': greenness_threshold,
                           'brightness_threshold': brightness_threshold}

    # For each band, compute tasseled cap band and add to output dataset
    for tc_band in tc_bands:
        # Create xarray of coefficient values used to multiply each band of input
        coeff = xr.Dataset(analysis_coefficient[tc_band])
        sensor_coeff = sensor_data * coeff
        # Sum all bands
        output_array[tc_band] = sensor_coeff.blue + sensor_coeff.green + \
            sensor_coeff.red + sensor_coeff.nir + sensor_coeff.swir_1 + sensor_coeff.swir_2
        output_array[str(tc_band+'_thresholded')] = output_array[tc_band].where(
            output_array[tc_band] > analysis_thresholds[str(tc_band+'_threshold')])
        if drop_tc_bands:
            output_array = output_array.drop(tc_band)

    # If drop = True, remove original bands
    if drop:
        bands_to_drop = list(sensor_data.data_vars)
        output_array = output_array.drop(bands_to_drop)

    return output_array



def animated_timeseries_WIT(
    ds,
    df,
    output_path,
    width_pixels=1000,
    interval=200,
    bands=["red", "green", "blue"],
    percentile_stretch=(0.02, 0.98),
    image_proc_func=None,
    title=False,
    show_date=True,
    annotation_kwargs={},
    onebandplot_cbar=True,
    onebandplot_kwargs={},
    shapefile_path=None,
    shapefile_kwargs={},
    pandasplot_kwargs={},
    time_dim="time",
    x_dim="x",
    y_dim="y",
):

    ###############
    # Setup steps #
    ###############

    # Test if all dimensions exist in dataset
    if time_dim in ds and x_dim in ds and y_dim in ds:

        # Test if there is one or three bands, and that all exist in both datasets:
        if ((len(bands) == 3) | (len(bands) == 1)) & all(
            [(b in ds.data_vars) for b in bands]
        ):

            # Import xarrays as lists of three band numpy arrays
            imagelist, vmin, vmax = _ds_to_arrraylist(
                ds,
                bands=bands,
                time_dim=time_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                percentile_stretch=percentile_stretch,
                image_proc_func=image_proc_func,
            )

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
            onebandplot_kwargs = dict(
                {
                    "cmap": "Greys",
                    "interpolation": "bilinear",
                    "vmin": vmin,
                    "vmax": vmax,
                    "tick_colour": "black",
                    "tick_fontsize": 11,
                },
                **onebandplot_kwargs,
            )

            # Use pop to remove the two special tick kwargs from the onebandplot_kwargs dict, and save individually
            onebandplot_tick_colour = onebandplot_kwargs.pop("tick_colour")
            onebandplot_tick_fontsize = onebandplot_kwargs.pop("tick_fontsize")

            # Set up annotation parameters that control font etc. The nested dict structure sets default
            # values which can be overwritten/customised by the manually specified `annotation_kwargs`
            annotation_kwargs = dict(
                {
                    "xy": (1, 1),
                    "xycoords": "axes fraction",
                    "xytext": (-5, -5),
                    "textcoords": "offset points",
                    "horizontalalignment": "right",
                    "verticalalignment": "top",
                    "fontsize": 15,
                    "color": "white",
                    "path_effects": [
                        PathEffects.withStroke(linewidth=3, foreground="black")
                    ],
                },
                **annotation_kwargs,
            )

            # Define default plotting parameters for the overlaying shapefile(s). The nested dict structure sets
            # default values which can be overwritten/customised by the manually specified `shapefile_kwargs`
            shapefile_kwargs = dict(
                {"linewidth": 2, "edgecolor": "black", "facecolor": "#00000000"},
                **shapefile_kwargs,
            )

            # Define default plotting parameters for the right-hand line plot. The nested dict structure sets
            # default values which can be overwritten/customised by the manually specified `pandasplot_kwargs`
            pandasplot_kwargs = dict({}, **pandasplot_kwargs)

            ###################
            # Initialise plot #
            ###################

            # Set up figure
            fig, (ax1, ax2) = plt.subplots(
                ncols=2, gridspec_kw={"width_ratios": [1, 2]}
            )
            fig.subplots_adjust(left=0, bottom=0, right=1,
                                top=1, wspace=0.2, hspace=0)
            fig.set_size_inches(10.0, height * 0.5, forward=True)
            ax1.axis("off")
            ax2.margins(x=0.01)
            ax2.xaxis.label.set_visible(False)

            # Initialise axesimage objects to be updated during animation, setting extent from dims
            extents = [
                float(ds[x_dim].min()),
                float(ds[x_dim].max()),
                float(ds[y_dim].min()),
                float(ds[y_dim].max()),
            ]
            im = ax1.imshow(imagelist[0], extent=extents, **onebandplot_kwargs)

            # Initialise right panel and set y axis limits
            # set up color palette
            pal = [
                sns.xkcd_rgb["cobalt blue"],
                sns.xkcd_rgb["neon blue"],
                sns.xkcd_rgb["grass"],
                sns.xkcd_rgb["beige"],
                sns.xkcd_rgb["brown"],
            ]

            # make a stacked area plot
            ax2.stackplot(
                df.index,
                df.wofs_area_percent,
                df.wet_percent,
                df.green_veg_percent,
                df.dry_veg_percent,
                df.bare_soil_percent,
                labels=["open water", "wet",
                        "green veg", "dry veg", "bare soil"],
                colors=pal,
                alpha=0.6,
                **pandasplot_kwargs,
            )

            ax2.legend(loc="lower left", framealpha=0.6)

            df1 = pd.DataFrame(
                {
                    "wofs_area_percent": df.wofs_area_percent,
                    "wet_percent": df.wofs_area_percent + df.wet_percent,
                    "green_veg_percent": df.wofs_area_percent
                    + df.wet_percent
                    + df.green_veg_percent,
                    "dry_veg_percent": df.wofs_area_percent
                    + df.wet_percent
                    + df.green_veg_percent
                    + df.dry_veg_percent,
                    "bare_soil_percent": df.dry_veg_percent
                    + df.green_veg_percent
                    + df.wofs_area_percent
                    + df.wet_percent
                    + df.bare_soil_percent,
                }
            )
            df1 = df1.set_index(df.index)

            line_test = df1.plot(
                ax=ax2, legend=False, colors="black", **pandasplot_kwargs
            )

            # set axis limits to the min and max
            ax2.set(xlim=(df.index[0], df.index[-1]), ylim=(0, 100))

            # add a legend and a tight plot box

            ax2.set_title("Fractional Cover, Wetness, and Water")

            # Initialise annotation objects to be updated during animation
            t = ax1.annotate("", **annotation_kwargs)

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
                _add_colourbar(
                    ax1,
                    im,
                    tick_fontsize=onebandplot_tick_fontsize,
                    tick_colour=onebandplot_tick_colour,
                    vmin=onebandplot_kwargs["vmin"],
                    vmax=onebandplot_kwargs["vmax"],
                )

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
                    ts = ds[time_dim][{time_dim: frame_i}].dt
                    year = ts.year.item()
                    month = ts.month.item()
                    day = ts.day.item()
                    date_string = "{} {} {}".format(
                        day, calendar.month_abbr[month], year
                    )

                except:

                    date_string = ds[time_dim][{
                        time_dim: frame_i}].values.item()

                # Create annotation string based on title and date specifications:
                title = title_list[frame_i]
                if title and show_date:
                    title_date = "{}\n{}".format(date_string, title)
                elif title and not show_date:
                    title_date = "{}".format(title)
                elif show_date and not title:
                    title_date = "{}".format(date_string)
                else:
                    title_date = ""

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
                    y = df1[
                        df1.index
                        <= datetime(year=year, month=month, day=day, hour=23, minute=59)
                    ].iloc[:, i]
                    x = df1[
                        df1.index
                        <= datetime(year=year, month=month, day=day, hour=23, minute=59)
                    ].index

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
            ani = animation.FuncAnimation(
                fig=fig,
                func=update_figure,
                frames=timesteps,
                interval=interval,
                blit=True,
            )

            # Export as either MP4 or GIF
            if output_path[-3:] == "mp4":
                print("    Exporting animation to {}".format(output_path))
                ani.save(output_path, dpi=width_pixels / 10.0)

            elif output_path[-3:] == "wmv":
                print("    Exporting animation to {}".format(output_path))
                ani.save(
                    output_path,
                    dpi=width_pixels / 10.0,
                    writer=animation.FFMpegFileWriter(
                        fps=1000 / interval, bitrate=4000, codec="wmv2"
                    ),
                )

            elif output_path[-3:] == "gif":
                print("    Exporting animation to {}".format(output_path))
                ani.save(output_path, dpi=width_pixels /
                         10.0, writer="imagemagick")

            else:
                print("    Output file type must be either .mp4, .wmv or .gif")

        else:
            print(
                "Please select either one or three bands that all exist in the input dataset"
            )

    else:
        print(
            "At least one x, y or time dimension does not exist in the input dataset. Please use the `time_dim`,"
            "`x_dim` or `y_dim` parameters to override the default dimension names used for plotting"
        )


# Define function to convert xarray dataset to list of one or three band numpy arrays


def _ds_to_arrraylist(
    ds, bands, time_dim, x_dim, y_dim, percentile_stretch, image_proc_func=None
):
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
            img_toshow = exposure.rescale_intensity(
                ds_i[bands[0]].values, in_range=(
                    p_low, p_high), out_range="image"
            )

        else:

            # Create new three band array
            rawimg = np.zeros((y, x, 3), dtype=np.float32)

            # Add xarray bands into three dimensional numpy array
            for band, colour in enumerate(bands):

                rawimg[:, :, band] = ds_i[colour].values

            # Stretch contrast using percentile values
            img_toshow = exposure.rescale_intensity(
                rawimg, in_range=(p_low, p_high), out_range=(0, 1.0)
            )

            # Optionally image processing
            if image_proc_func:

                img_toshow = image_proc_func(img_toshow).clip(0, 1)

        array_list.append(img_toshow)

    return array_list, p_low, p_high


def _add_colourbar(
    ax, im, vmin, vmax, cmap="Greys", tick_fontsize=15, tick_colour="black"
):
    """
    Add a nicely formatted colourbar to an animation panel
    """

    # Add colourbar
    axins2 = inset_axes(ax, width="97%", height="4%", loc=8, borderpad=1)
    plt.gcf().colorbar(
        im, cax=axins2, orientation="horizontal", ticks=np.linspace(vmin, vmax, 3)
    )
    axins2.xaxis.set_ticks_position("top")
    axins2.tick_params(axis="x", colors=tick_colour, labelsize=tick_fontsize)

    # Justify left and right labels to edge of plot
    axins2.get_xticklabels()[0].set_horizontalalignment("left")
    axins2.get_xticklabels()[-1].set_horizontalalignment("right")
    labels = [item.get_text() for item in axins2.get_xticklabels()]
    labels[0] = "  " + labels[0]
    labels[-1] = labels[-1] + "  "


if __name__ == "__main__":
    # print that we are running the testing
    print("Testing..")
    # import doctest to test our module for documentation
    import doctest

    doctest.testmod()
    print("Testing done")
