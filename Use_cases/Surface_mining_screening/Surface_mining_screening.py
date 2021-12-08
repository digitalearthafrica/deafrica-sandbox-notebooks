## Surface_mining_screening.py
'''
Description: This file contains a set of python functions for the Surface_mining_screening notebook

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

'''

# Import required packages
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import datacube
from datacube.utils import geometry
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib._color_data as mcd
from matplotlib.colors import ListedColormap

from deafrica_tools.datahandling import load_ard
from deafrica_tools.bandindices import calculate_indices, dualpol_indices
from deafrica_tools.plotting import rgb, map_shapefile
from deafrica_tools.spatial import xr_rasterize, xr_vectorize


def calculate_area_per_pixel(resolution):
    """
    Takes a resolution in metres and return the area of that
    pixel in square kilometres.
    """
    
    pixel_length = resolution  # in metres
    m_per_km = 1000  # conversion from metres to kilometres
    area_per_pixel = pixel_length**2 / m_per_km**2
    return area_per_pixel

def load_vector_file(vector_file):
    """
    Takes a vector file and returns the attributes and geometry
    """
    
    # Determine file extension
    extension = vector_file.split(".")[-1]

    if extension == 'kml':

        # Enable fiona driver to read KML
        gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
        # Read KML file into GeoDataFrame
        gdf = gpd.read_file(vector_file, driver='KML')

    else:
        # Read vector file into GeoDataFrame
        gdf = gpd.read_file(vector_file)

    # Get geometry from GeoDataFrame
    geom = geometry.Geometry(gdf.unary_union, gdf.crs)
    
    # Plot the vector on an interactive map
    map_shapefile(gdf, attribute=gdf.columns[0], fillOpacity=0, weight=3)

    return gdf, geom

def process_data(gdf, geom, start_year, end_year, product='s2'):
    """
    For a given geometry, load data for a given baseline and analysis year. 
    Data loaded is Sentinel-2 geomedian, and WOfS
    """
    #connect to datacube
    dc = datacube.Datacube(app='surface_mining')
    
    query = {
        'geopolygon' : geom,    
    }
    
    if product=='s1':
        # Load available data from Sentinel-1
        ds = load_ard(
            dc=dc,
            products=['s1_rtc'],
            time=(f'{start_year}', f'{end_year}'),
            measurements = ['vv', 'vh'],
            resolution = (-20, 20),
            output_crs='epsg:6933',
            group_by='solar_day',
            **query,
        )
        # VH/VV will help create an RGB image
        ds['vh/vv'] = ds.vh/ds.vv
        
        # calculate veg index
        ds = dualpol_indices(ds, index='RVI')
        ds = ds.resample(time='1Y').median('time')
        
    if product =='s2':
        
        #load gm
        ds = dc.load(product='gm_s2_annual',
                         measurements=["red","green","blue","nir"],
                         time=(f'{start_year}', f'{end_year}'),
                         resolution = (-10, 10),
                         **query)
        
        # For loaded Sentinel-2 Geomedian data, compute the annual geomedian and Calcute NDVI
        ds = calculate_indices(ds, ['NDVI'], collection='s2')
    
    #load wofs summary
    ds_wofs = dc.load(product="wofs_ls_summary_annual",
                      time=(f'{start_year}', f'{end_year}'),
                      resampling='nearest',
                      like=ds.geobox
                      ).frequency


    # Convert the polygon to a raster that matches our imagery data
    mask = xr_rasterize(gdf, ds)

    # Mask dataset to set pixels outside the polygon to `NaN`
    ds = ds.where(mask)
    
    if product=='s2':
        rgb(ds, col='time',col_wrap=len(ds.time.values))
    if product=='s1':
        med_s1 = ds[['vv','vh','vh/vv']].median()
        rgb(ds[['vv', 'vh', 'vh/vv']]/med_s1,
            bands=['vv','vh', 'vh/vv'],
            col='time',
            col_wrap=len(ds.time.values))
    
    # Interested in observations where water has appeared in at least 10% of observations
    water_frequency_boolean = xr.where(ds_wofs > 0.1, True, False)
    
    # Total the number of times water was observed over the dataset
    water_frequency_sum = water_frequency_boolean.sum('time')
    water_frequency_sum = water_frequency_sum.where(mask)
            
    return ds, water_frequency_sum

def calculate_vegetation_loss(ds, product='s2', threshold=-0.15):
    """
    Calculate vegetaion loss between each year.
    Takes an xarray dataset, which must have NDVI as an array
    """
    if product=='s2':
        index='NDVI'
        
    if product=='s1':
        index='RVI'
    
    # Determine the change by shifting the array by a year and taking the difference
    ds_shift = ds[index].shift(time=1)
    ds_change_ndvi = ds[index] - ds_shift
     
    # Define loss as where the change is less than the threshold
    if threshold == 'otsu':
        threshold = threshold_otsu(ds_change_ndvi.fillna(0).values)
    
    vegetation_loss = ds_change_ndvi < threshold
    vegetation_loss = vegetation_loss.where(vegetation_loss == True)
    
    # Determine the total area that experienced vegetation loss each year
    count_vegetation_loss = vegetation_loss.count(dim=['x', 'y'])
    vegetation_loss_area = count_vegetation_loss * calculate_area_per_pixel(ds.x.resolution)

    plt.figure(figsize=(11, 4))
    vegetation_loss_area.plot.line('g-o')
    plt.grid()
    plt.title('Annual Vegetation Loss')
    plt.ylabel('Area of vegetation loss (km sq)')
    
    # Determine all areas that experienced any vegetation loss over all years
    vegetation_loss_sum = vegetation_loss.sum('time')
    
    return vegetation_loss, vegetation_loss_sum


def plot_possible_mining(ds, vegetation_loss_sum, water_frequency_sum, product='s2', plot_filename="./results/Possible_Mining.png"):
    
    if product=='s2':
        index='NDVI'
        
    if product=='s1':
        index='RVI'
        
    # Select the first year's NDVI as the background image
    background_image = ds[index].isel(time=0)
    
    # Determine the mining areas: vegetation loss & one year of water occurance
    mining_area = vegetation_loss_sum.where(water_frequency_sum == True)
    mining_area = xr.where(mining_area >= 0, 1, 0)
    
    # Vectorize and buffer the mining area
    mining_area_vector = xr_vectorize(mining_area,
                                      mask=mining_area.values == 1,
                                      crs='EPSG:'+str(ds.geobox.crs.to_epsg())
                                 )
    mining_area_vector_buffer = mining_area_vector.buffer(90)
    
    # Rasterize the buffered area
    mining_area_buffer = xr_rasterize(gdf=mining_area_vector_buffer,
                                      da=mining_area,
                                      crs='EPSG:'+str(ds.geobox.crs.to_epsg())
                                     )
    
    # Find all vegetation loss within the buffer, and check if these are also mining areas
    vegetation_loss_buffer = vegetation_loss_sum.where(mining_area_buffer == True)
    vegetation_loss_buffer = xr.where(vegetation_loss_buffer > 0, True, False)
    vegetation_loss_buffer = vegetation_loss_buffer.where(vegetation_loss_buffer == True)
    
    water_observed_buffer = water_frequency_sum.where(mining_area_buffer == True)
    water_observed_buffer = xr.where(water_observed_buffer > 0, True, False)
    water_observed_buffer =water_observed_buffer.where(water_observed_buffer == True)
    
    # Construct and save the figure
    plt.figure(figsize=(12, 12))
    background_image.plot.imshow(cmap='Greys', add_colorbar=False)
    vegetation_loss_buffer.plot.imshow(cmap=ListedColormap(['Gold']), add_colorbar=False)
    water_observed_buffer.plot.imshow(cmap=ListedColormap(['Gold']), add_colorbar=False)

    plt.legend(
            [Patch(facecolor='Gold')], 
            ['Possible Mining Site'],
            loc = 'upper left'
        )

    plt.title('Possible Mining Areas')

    plt.savefig(plot_filename)
    
    return vegetation_loss_buffer

def plot_vegetationloss_mining(ds, vegetation_loss, vegetation_loss_buffer, product='s2'):
    
    if product=='s2':
        index='NDVI'
        
    if product=='s1':
        index='RVI'
    
    year_arr = (pd.to_datetime(vegetation_loss.time.values)).year
    background_image = ds[index].isel(time=0)
    
    total = np.count_nonzero(~np.isnan(background_image.values)) * calculate_area_per_pixel(ds.x.resolution)
    
    print(pd.DataFrame([total], index=['Total Area(kmsq) of the vector file'], columns=['']))
    print('...................................................................')
    
    vegetation_loss_mininig = vegetation_loss.where((vegetation_loss == True) & (vegetation_loss_buffer == True))
    total_vegetation_loss_mininig = vegetation_loss_mininig.count(dim=['x', 'y'])
    total_vegetation_loss = vegetation_loss.where(vegetation_loss==True).count(dim=['x', 'y'])

    vegetation_loss_mininig_area = total_vegetation_loss_mininig * calculate_area_per_pixel(ds.x.resolution)
    vegetation_loss_area = total_vegetation_loss * calculate_area_per_pixel(ds.x.resolution)

    print(pd.DataFrame(data=[vegetation_loss_area.values, 
                  (vegetation_loss_area.values/total)*100,
                  vegetation_loss_mininig_area.values,
                  (vegetation_loss_mininig_area.values/total)*100
                 ], 
                 columns=[str(i) for i in year_arr], 
                 index=['Any Vegetation Loss(kmsq)',
                        'Any Vegetation Loss(%)',
                        'Vegetation Loss from Possible Mining(kmsq)',
                        'Vegetation Loss from Possible Mining(%)']
                ))
    
    print('...................................................................')
    # Construct and save the figure
    plt.figure(figsize=(12, 12))
    
    
    vegetation_loss_area.plot.line('g-o', figsize=(11, 4))
    vegetation_loss_mininig_area.plot.line('r-^')
   
    plt.legend(
            [Patch(facecolor='Green'), Patch(facecolor='Red')], 
            ['Any Vegetation Loss', 'Vegetation Loss from Possible Mining'],
            loc = 'upper left'
        )
    
  
    plt.grid()
    plt.title('Vegetation Loss')
    plt.ylabel('Area : km sq')
    plt.show();
    
    print('...................................................................')
    
    size_n = vegetation_loss.time.size
    color_scheme = [name for name in mcd.TABLEAU_COLORS][:size_n]

    fig, axes = plt.subplots(1,2, figsize=(20,10))
    color_array = []
    year_array = []
    background_image.plot.imshow(cmap='Greys', add_colorbar=False)
    
    
    for i in range(1, size_n):
        vegetation_loss_mininig.isel(time=i).plot.imshow(ax=axes[1],add_colorbar=False, cmap=ListedColormap([color_scheme[i]]))
        color_array.append(Patch(facecolor=f'{color_scheme[i]}'))
        year_array.append(f'{year_arr[i]}')
        
    if product=='s2':
        rgb(ds, index=[-1], ax=axes[0])
    if product=='s1':
        med_s1 = ds[['vv','vh','vh/vv']].median()
        rgb(ds[['vv', 'vh', 'vh/vv']]/med_s1,
            bands=['vv','vh', 'vh/vv'],
            index=[-1], ax=axes[0])

    axes[0].set_title('RGB plot from most recent composite image')

    axes[1].legend(color_array, year_array, loc = 'upper left')

    plt.title(f'Vegetation Loss from Possible Mining from {year_arr[0]} to {year_arr[-1]}')
    plt.show()

