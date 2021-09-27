# notebookapp_wetlands.py
'''
This file contains functions for creating an interactive map for 
selecting a polygon and running the wetlands insight tool.

Available functions:
    select_region_app
    WIT_app

Last modified: September 2021
'''

# Load modules
import datacube
import seaborn as sns
import matplotlib.pyplot as plt
from odc.ui import select_on_a_map
from datacube.utils.geometry import CRS
from ipyleaflet import (WMSLayer, basemaps, basemap_to_tiles)
from traitlets import Unicode

import sys
sys.path.insert(1, '../Real_world_examples/')
from wetlands import WIT_drill

# from deafrica_tools.spatial import reverse_geocode
from deafrica_tools.dask import create_local_dask_cluster

def select_region_app(date,
                      product,
                      size_limit=15000):
    '''
    An interactive app that allows the user to select a region from a
    map using imagery from Sentinel-2 and Landsat. The output of this
    function is used as the input to `export_image_app` to export high-
    resolution satellite images.
    
    Last modified: September 2021

    Parameters
    ----------
    date : str
        The exact date used to plot imagery on the interactive map
        (e.g. `date='1988-01-01'`). 
    product : str
        The satellite data to plot on the interactive map. Three options
        are currently supported:
            "Landsat": data from the Landsat 5, 7 and 8 satellites
            "Sentinel-2": data from Sentinel-2A and Sentinel-2B
    size_limit : int, optional
        An optional size limit for the area selection in sq km.
        Defaults to 10000 sq km.
        
    Returns
    -------
    A dictionary containing 'geopolygon' (defining the area to 
    export imagery from), 'date' (date used to export imagery), and 
    'satellites' (the satellites from which to extract imagery). These
    are passed to the `export_image_app` function to export the image.
    '''

    ########################
    # Select and load data #
    ########################

    # Load DEA WMS
    class TimeWMSLayer(WMSLayer):
        time = Unicode('').tag(sync=True, o=True)

    time_wms = TimeWMSLayer(url='https://ows.digitalearth.africa/',
                            layers=product,
                            time=date,
                            format='image/png',
                            transparent=True,
                            attribution='Digital Earth Africa')

    # Plot interactive map to select area
    basemap = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    geopolygon = select_on_a_map(height='1000px',
                                 layers=(
                                     basemap,
                                     time_wms,
                                 ),
                                 center=(4, 20),
                                 zoom=4)

    # Test size of selected area
    area = geopolygon.to_crs(crs=CRS('epsg:6933')).area / 1000000
    if area > size_limit:
        print(f'Warning: Your selected area is {area:.00f} sq km. '
              f'Please select an area of less than {size_limit} sq km.'
              f'\nTo select a smaller area, re-run the cell '
              f'above and draw a new polygon.')

    else:
        return geopolygon

    
def WIT_app(gdf,
            time_range,
            min_gooddata=0.80,
            TCW_threshold=-0.035,
            export_csv=None,
            export_plot=None,
            dask_chunks=dict(x=1000, y=1000, time=1),
            verbose=False
):
    
    '''
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or datacube.utils.geometry._base.Geometry
        The geometry over which to calculate WIT
    time : tuple
        a tuple containing the time range over which to run the WIT.
        e.g. ('2015-01' , '2019-12')
    min_gooddata : Float, optional
        A number between 0 and 1 (e.g 0.8) indicating the minimum percentage
        of good quality pixels required for a satellite observation to be loaded
        and therefore included in the WIT plot.  Defaults to 0.8, which should
        be considered a minimum percentage.
    TCW_threshold : float, optional
        The tasseled cap wetness threshold, beyond which a pixel will be
        considered 'wet'. Defaults to -0.035. 
    export_csv : str, optional
        To export the returned pandas dataframe provide a string 
        output file name and path (e.g. 'output/results.csv')
    export_plot : str, optional
        Option to export the stack plot as .png, provide a string 
        output file name and path (e.g. 'output/results.png')
    dask_chunks : dict, optional
        To lazily load the datasets using dask, pass a dictionary containing
        the dimensions over which to chunk e.g. {'time':1, 'x':500, 'y':500}.
     verbose: bool, optional
         If true print statements will be output at each step of the analysis
         
    Returns
    -------
        A stackplot of fractional cover, water, and wetness summarized over 
        the input geometry
   
    '''
    
    # Connect to datacube database
    dc = datacube.Datacube(app='wetland_app')

    # Configure local dask cluster
    create_local_dask_cluster()
    
    #run wetlands polygon drill
    df = WIT_drill(gdf=gdf,
                    time=time_range,
                    min_gooddata=min_gooddata,
                    TCW_threshold=-TCW_threshold,
                    export_csv=export_csv,
                    dask_chunks=dask_chunks,
                    verbose=verbose,
                       )
    
    # save the csv
    if export_csv:
        print("exporting csv: " + export_csv)
        df.to_csv(export_csv, index_label="Datetime")
    
    #---Plotting------------------------------
 
    fontsize = 17
    #set up color palette
    pal = [sns.xkcd_rgb["cobalt blue"],
           sns.xkcd_rgb["neon blue"],
           sns.xkcd_rgb["grass"],
           sns.xkcd_rgb["beige"],
           sns.xkcd_rgb["brown"]]       

    #make a stacked area plot
    plt.clf()
    fig=plt.figure(figsize = (22,6))
    plt.stackplot(df.index, 
                  df.wofs_area_percent, 
                  df.wet_percent, 
                  df.green_veg_percent, 
                  df.dry_veg_percent,
                  df.bare_soil_percent,
                  labels=['open water',
                          'wet',
                          'green veg',
                          'dry veg',
                          'bare soil',
                         ], colors=pal, alpha = 0.6)

    #set axis limits to the min and max
    plt.axis(xmin = df.index[0], xmax = df.index[-1], ymin = 0, ymax = 100)
    plt.tick_params(labelsize=fontsize)
    #add a legend and a tight plot box
    plt.legend(loc='lower left', framealpha=0.6)
    plt.title('Fractional Cover, Wetness, and Water')
    plt.tight_layout();
    if export_plot:
        #save the figure
        plt.savefig(f'{export_csv[:-3]}_WIT.png')
        
    return df