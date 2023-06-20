## long_term_water_extent.py
"""
Description: This file contains a set of python functions for the Long_term_water_extent notebook

License: The code in this file is licensed under the Apache License,
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

"""

# Import required packages

# Force GeoPandas to use Shapely instead of PyGEOS
# In a future release, GeoPandas will switch to using Shapely by default.
import os
os.environ['USE_PYGEOS'] = '0'

import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.filters import threshold_li
from matplotlib.patches import Patch
from datacube.utils import geometry
from datacube.utils import masking
from deafrica_tools.spatial import xr_rasterize

# Turn off  RuntimeWarning: divide by zero or RuntimeWarning: invalid value warnings.
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def get_geometry(gdf):
    """Get geometry for use with datacube from geopandas GeoDataFrame."""
    gdf_crs = f"EPSG:{gdf.crs.to_epsg()}"
    gdf_geom = gdf.loc[0, "geometry"].__geo_interface__

    geom = geometry.Geometry(gdf_geom, gdf_crs)
    return geom


def load_vector_file(file):
    """Load geopandas GeoDataFrame from vector file and get geometry"""
    vector = gpd.read_file(file)
    vector_geom = get_geometry(vector)

    return vector, vector_geom


def get_resampled_labels(ds, freq, date_format="%b %y"):
    """Get the date-range for each resampling period as a list of labels."""

    # Get left label
    left = ds.resample(time=freq, label="left").groups
    left_str = [pd.to_datetime(str(key)).strftime(date_format) for key in left]

    # Get right label
    right = ds.resample(time=freq, label="right").groups
    right_str = [pd.to_datetime(str(key)).strftime(date_format) for key in right]

    # Create combined label
    pairs = zip(left_str, right_str)
    labels = [f"{l} - {r}" for l, r in pairs]

    return labels


def resample_water_observations(ds, freq, radar=False, date_format="%b %y"):
    """For loaded wofs data, resample and calculate waterbody area"""

    if radar == False:
        # Mask for water
        ds = masking.make_mask(ds, wet=True)
    
    resampled_ds = ds.resample(time=freq, label="left").max().compute()
    resampled_labels = get_resampled_labels(ds, freq, date_format)
    
    #determine a threshold for Radar water index 
    if radar == True:
        threshold = threshold_li(resampled_ds.swi.values)
        print('Automatic SWI threshold: '+str(round(threshold, 2)))
        resampled_ds = xr.where(resampled_ds > threshold, 1, 0)
    
    # Calculate area of water
    pixel_length = ds.x.values[1] - ds.x.values[0]  # in metres
    m_per_km = 1000  # conversion from metres to kilometres
    area_per_pixel = pixel_length ** 2 / m_per_km ** 2

    resampled_area_ds = resampled_ds.sum(dim=["x", "y"]) * area_per_pixel
    
    if radar == False:
        return resampled_ds, resampled_area_ds.water
    else:
        return resampled_ds, resampled_area_ds.swi
    


def resample_rainfall_observations(ds, freq, mask):
    """For loaded rainfall data, calculate average spatial for each time, then total over resampling period."""

    # create mask
    mask = xr_rasterize(mask, ds)

    # mask data
    masked_ds = ds.where(mask)

    # Calculate the average rainfall over the catchment at each time step
    average_rainfall_ds = masked_ds.mean(dim=("x", "y"))

    # Calculate the total rainfall over each resampling period
    average_rainfall_resampled_ds = average_rainfall_ds.resample(
        time=freq, label="right"
    ).sum(dim="time").compute()

    return average_rainfall_resampled_ds.rainfall


def compare_extent_and_rainfall(water_ds, rainfall_ds, rainfall_units, labels):
    """Create a combined plot of water extent and rainfall"""
    
    # plot daily total precipitation for this area
    fig, ax1 = plt.subplots(figsize=(15, 5))

    plt.xticks(rotation=65)

    # Create histogram of rainfall
    ax1.bar(
        labels,
        rainfall_ds.values,
        color="lightblue",
        label="Rainfall",
    )

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()

    # Create line chart of water area
    ax2 = ax1.twinx()
    ax2.plot(
        labels,
        water_ds.values,
        color="red",
        marker="^",
        markersize=4,
        linewidth=1,
        label="Waterbody Area",
    )

    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()

    # Format plot
    fig.suptitle(
        "Evolution of waterbody surface area compared to catchment rainfall (CHIRPS)"
    )

    ax1_handles.extend(ax2_handles)
    ax1_labels.extend(ax2_labels)
    ax1.legend(ax1_handles, ax1_labels, loc="upper left")

    ax1.set_ylabel(f"Total Precipitation ({rainfall_units})")

    ax2.set_ylabel("Waterbody area (km$^2$)")
    ax2.tick_params(axis="y", colors="red")
    ax2.yaxis.label.set_color("red")

    return fig


def calculate_change_in_extent(start_date, end_date, ds, radar=False):
    """Create a plot showing areas where water has appeared or disappeared between two dates."""

    baseline_ds = ds.sel(time=start_date, method="nearest")
    analysis_ds = ds.sel(time=end_date, method="nearest")
    if radar == True:
        compare = ds.swi.sel(time=[baseline_ds.time.values, analysis_ds.time.values])
    else:
        compare = ds.water.sel(time=[baseline_ds.time.values, analysis_ds.time.values])

    # The two period Extract the two periods(Baseline and analysis) dataset from
    analyse_total_value = compare.isel(time=1).astype(int)
    change = analyse_total_value - compare.isel(time=0).astype(int)

    water_appeared = change.where(change == 1)
    permanent_water = change.where((change == 0) & (analyse_total_value == 1))
    permanent_land = change.where((change == 0) & (analyse_total_value == 0))
    water_disappeared = change.where(change == -1)

    pixel_length = ds.x.values[1] - ds.x.values[0]  # in metres
    m_per_km = 1000  # conversion from metres to kilometres
    area_per_pixel = pixel_length ** 2 / m_per_km ** 2

    total_area = analyse_total_value.count().values * area_per_pixel
    water_apperaed_area = water_appeared.count().values * area_per_pixel
    permanent_water_area = permanent_water.count().values * area_per_pixel
    water_disappeared_area = water_disappeared.count().values * area_per_pixel

    # Produce plot
    dark_green = "#33a02c"
    light_green = "#b2df8a"
    dark_blue = "#1f78b4"
    light_blue = "#a6cee3"

    water_appeared_color = dark_blue
    water_disappeared_color = dark_green
    stable_color = light_blue
    land_color = light_green
    
    x_pix = baseline_ds.dims['x']
    y_pix = baseline_ds.dims['y']

    x_inch = 6
    y_inch = y_pix * (x_inch/x_pix)
    fig, ax = plt.subplots(1, 1, figsize=(x_inch, y_inch))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0*(x_pix/y_pix), adjustable='box')

    permanent_land.plot.imshow(
        cmap=ListedColormap([land_color]),
        add_colorbar=False,
        add_labels=False,
        ax=ax,
    )

    water_appeared.plot.imshow(
        cmap=ListedColormap([water_appeared_color]),
        add_colorbar=False,
        add_labels=False,
        ax=ax,
    )
    water_disappeared.plot.imshow(
        cmap=ListedColormap([water_disappeared_color]),
        add_colorbar=False,
        add_labels=False,
        ax=ax,
    )

    permanent_water.plot.imshow(
        cmap=ListedColormap([stable_color]),
        add_colorbar=False,
        add_labels=False,
        ax=ax,
    )

    plt.legend(
        [
            Patch(facecolor=land_color),
            Patch(facecolor=stable_color),
            Patch(facecolor=water_disappeared_color),
            Patch(facecolor=water_appeared_color),
        ],
        [
            f"Land present in both",
            f"Water present in both",
            f"Water disappeared: {round(water_disappeared_area, 2)} km$^2$",
            f"Water appeared: {round(water_apperaed_area, 2)} km$^2$",
        ],
        loc="upper right",
    )

    plt.title("Change in water extent: " + str(baseline_ds.time.values)[0:10] + " to " + str(analysis_ds.time.values)[0:10])

    return fig
