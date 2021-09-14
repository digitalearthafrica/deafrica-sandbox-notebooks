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
from datacube.utils import geometry
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib._color_data as mcd
from matplotlib.colors import ListedColormap

from odc.algo import xr_geomedian
from deafrica_tools.datahandling import load_ard, mostcommon_crs, wofs_fuser
from deafrica_tools.bandindices import calculate_indices
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

def process_data(dc, gdf, geom, start_year, end_year):
    """
    For a given geometry, load data for a given baseline and analysis year. 
    Data loaded is Sentinel-2, which is converted to a geomedian, and 
    Water Observations from Space (WOfS)
    """

    #Create a query object
    product = 'gm_s2_annual'

    query = {
        'group_by': 'solar_day',
        'geopolygon' : geom,    
    }
    
    # Identify the most common projection system in the input query
    output_crs = mostcommon_crs(dc=dc, product=product, query=query)

    ds_geomedian = dc.load(product=product,
                     output_crs=output_crs,
                     measurements=["red","green","blue","nir"],
                     time=(f'{start_year}', f'{end_year}'),
                     resolution = (-10, 10),
                     **query)
    
    ds_wofs = dc.load(product="wofs_ls_summary_annual",
                      time=(f'{start_year}', f'{end_year}'),
                      output_crs=output_crs,
                      resolution = (-10, 10),
                      **query)

    # For loaded Sentinel-2 Geomedian data, compute the annual geomedian and Calcute NDVI
    ds_geomedian = calculate_indices(ds_geomedian, ['NDVI'], collection='s2')

    # Convert the polygon to a raster that matches our imagery data
    mask = xr_rasterize(gdf, ds_geomedian)

    # Mask dataset to set pixels outside the polygon to `NaN`
    ds_geomedian = ds_geomedian.where(mask)
    
    # Find the middle year and plot the geomedian for the first, middle and last years
    try:
        mid = int(round(ds_geomedian.time.size / 2, 0))
        rgb(ds_geomedian, index_dim='time', index=[0, mid, -1])
    except:
        rgb(ds_geomedian, index_dim='time', index=[0, -1])
    
    # Process WOfS data to 
    count_wet=ds_wofs.count_wet.where(ds_wofs.count_wet>=0)
    count_clear=ds_wofs.count_clear.where(ds_wofs.count_clear>=0)
    water_frequency=count_wet/count_clear
    water_frequency = water_frequency.compute()
    # Interested in observations where water has appeared in at least 10% of observations
    water_frequency_boolean = xr.where(water_frequency > 0.1, True, False)
    
    # Total the number of times water was observed over the dataset
    water_frequency_sum = water_frequency_boolean.sum('time')
    water_frequency_sum = water_frequency_sum.where(mask)
            
    return ds_geomedian, water_frequency_sum, output_crs

def calculate_vegetation_loss(ds):
    """
    Calculate vegetaion loss between each year.
    Takes an xarray dataset, which must have NDVI as an array
    """
    
    # Determine the change by shifting the array by a year and taking the difference
    ds_shift = ds.NDVI.shift(time=1)
    ds_change_ndvi = ds.NDVI - ds_shift
    
    # Determine a threshold to judge if vegetation has increased or decreased
    threshold = threshold_otsu(ds_change_ndvi.fillna(0).values)
    
    # Define loss as where the change is less than the threshold
    vegetation_loss = ds_change_ndvi < threshold
    vegetation_loss = vegetation_loss.where(vegetation_loss == True)
    
    # Determine the total area that experienced vegetation loss each year
    ds_resolution = ds.x.resolution # metres per pixel
    count_vegetation_loss = vegetation_loss.count(dim=['x', 'y'])
    vegetation_loss_area = count_vegetation_loss * calculate_area_per_pixel(ds_resolution)

    plt.figure(figsize=(11, 4))
    vegetation_loss_area.plot.line('g-o')
    plt.grid()
    plt.title('Vegetation Loss')
    plt.ylabel('Area : km sq')
    
    # Determine all areas that experienced any vegetation loss over all years
    vegetation_loss_sum = vegetation_loss.sum('time')
    
    return vegetation_loss, vegetation_loss_sum


def plot_possible_mining(ds, vegetation_loss_sum, water_frequency_sum, crs, plot_filename="./results/Possible_Mining.png"):
    
    # Select the first year's NDVI as the background image
    background_image = ds.isel(time=0).NDVI
    
    # Determine the mining areas: vegetation loss & one year of water occurance
    mining_area = vegetation_loss_sum.where(water_frequency_sum == True)
    mining_area = xr.where(mining_area >= 0, 1, 0)
    
    # Vectorize and buffer the mining area
    mining_area_vector = xr_vectorize(mining_area,
                                      mask=mining_area.values == 1,
                                      crs=crs
                                 )
    mining_area_vector_buffer = mining_area_vector.buffer(90)
    
    # Rasterize the buffered area
    mining_area_buffer = xr_rasterize(gdf=mining_area_vector_buffer,
                                      da=mining_area,
                                      crs=crs
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

def plot_vegetationloss_mining(ds, vegetation_loss, vegetation_loss_buffer):
    
    year_arr = (pd.to_datetime(vegetation_loss.time.values)).year
    background_image = ds.isel(time =0).NDVI
    
    total = background_image.count(dim=['x', 'y']) * calculate_area_per_pixel(10)
    
    print(pd.DataFrame([total.values], index=['Total Area(kmsq) of the vector file'], columns=['']))
    print('...................................................................')
    
    vegetation_loss_mininig = vegetation_loss.where((vegetation_loss == True) & (vegetation_loss_buffer == True))
    
    total_vegetation_loss_mininig = vegetation_loss_mininig.count(dim=['x', 'y'])
    total_vegetation_loss = vegetation_loss.where(vegetation_loss==True).count(dim=['x', 'y'])

    vegetation_loss_mininig_area = total_vegetation_loss_mininig * calculate_area_per_pixel(10)
    vegetation_loss_area = total_vegetation_loss * calculate_area_per_pixel(10)
    
    print(pd.DataFrame([vegetation_loss_area.values, 
                  vegetation_loss_mininig_area.values,
                  '----',
                  (vegetation_loss_area.values / total.values) * 100 , 
                  (vegetation_loss_mininig_area.values / total.values) * 100
                 ], 
                 
                 columns=year_arr, 
                 index=['Any Vegetation Loss(kmsq)',
                        'Vegetation Loss from Possible Mining(kmsq)',
                        '',
                        'Any Vegetation Loss(%)', 
                        'Vegetation Loss from Possible Mining(%)']
                ))
    
    print('...................................................................')
    # Construct and save the figure
    plt.figure(figsize=(12, 12))
    
    
    vegetation_loss_area.plot.line('g-o', figsize=(11, 4))
    vegetation_loss_mininig_area.plot.line('r-^')
   
    plt.legend(
            [Patch(facecolor='Green'), Patch(facecolor='Red')], 
            ['Total Area of vector file', 'Any Vegetation Loss', 'Vegetation Loss from Possible Mining'],
            loc = 'upper left'
        )
    
  
    plt.grid()
    plt.title('Vegetation Loss')
    plt.ylabel('Area : km sq')
    plt.show()
    
    print('...................................................................')
    
    size_n = vegetation_loss.time.size
    color_scheme = [name for name in mcd.TABLEAU_COLORS][:size_n]

    plt.figure(figsize=(12, 12))
    color_array = []
    year_array = []
    background_image.plot.imshow(cmap='Greys', add_colorbar=False)
    
    
    for i in range(1, size_n):
        vegetation_loss_mininig.isel(time=i).plot.imshow(add_colorbar=False, cmap=ListedColormap([color_scheme[i]]))
        color_array.append(Patch(facecolor=f'{color_scheme[i]}'))
        year_array.append(f'{year_arr[i]}')
        
    


    plt.legend(color_array, year_array, loc = 'upper left')

    plt.title(f'Vegetation Loss from Possible Mining from {year_arr[0]} to {year_arr[-1]}')
    plt.show()

