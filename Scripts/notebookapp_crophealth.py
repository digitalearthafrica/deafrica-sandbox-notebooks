# notebookapp_crophealth.py
'''
This file contains functions for loading and interacting with data in the
crop health notebook, inside the Real_world_examples folder.

Available functions:
    load_crophealth_data
    run_crophelath_app

Last modified: January 2020
'''

# Load modules
from ipyleaflet import (
    Map,
    GeoJSON,
    DrawControl,
    basemaps
)
import datetime as dt
import datacube
from osgeo import ogr
import matplotlib as mpl
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import geometry_mask
import xarray as xr
from IPython.display import display
import warnings
import ipywidgets as widgets
import json
import geopandas as gpd
from io import BytesIO

# Load utility functions
from deafrica_datahandling import load_ard
from deafrica_spatialtools import xr_rasterize
from deafrica_bandindices import calculate_indices


def load_crophealth_data(lat, lon, buffer):
    """
    Loads Landsat 8 analysis-ready data (ARD) product for the crop health
    case-study area over the last two years.
    Last modified: April 2020
    
    Parameters
    ----------
    lat: float
        The central latitude to analyse
    lon: float
        The central longitude to analyse
    buffer:
         The number of square degrees to load around the central latitude and longitude. 
         For reasonable loading times, set this as `0.1` or lower.

    Returns
    ----------
    ds: xarray.Dataset 
        data set containing combined, masked data
        Masked values are set to 'nan'
    """
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Initialise the data cube. 'app' argument is used to identify this app
    dc = datacube.Datacube(app='Crophealth-app')
    
    # Define area to load
    latitude = (lat - buffer, lat + buffer)
    longitude = (lon - buffer, lon + buffer)

    # Specify the date range
    # Calculated as today's date, subtract 730 days to collect two years of data
    # Dates are converted to strings as required by loading function below
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=730)

    time = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    # Construct the data cube query
    products = ["ls8_usgs_sr_scene"]
    
    query = {
        'x': longitude,
        'y': latitude,
        'time': time,
                'measurements': [
            'red',
            'green',
            'blue',
            'nir',
            'swir2'
        ],
        'output_crs': 'EPSG:6933',
        'resolution': (-30, 30)
    }

    # Load the data and mask out bad quality pixels
    ds = load_ard(dc, products=products, min_gooddata=0.5, **query)

    # Calculate the normalised difference vegetation index (NDVI) across
    # all pixels for each image.
    # This is stored as an attribute of the data
    ds = calculate_indices(ds, index='NDVI', collection='s2')

    # Return the data
    return(ds)


def run_crophealth_app(ds, lat, lon, buffer):
    """
    Plots an interactive map of the crop health case-study area and allows
    the user to draw polygons. This returns a plot of the average NDVI value
    in the polygon area.
    Last modified: January 2020
    
    Parameters
    ----------
    ds: xarray.Dataset 
        data set containing combined, masked data
        Masked values are set to 'nan'
    lat: float
        The central latitude corresponding to the area of loaded ds
    lon: float
        The central longitude corresponding to the area of loaded ds
    buffer:
         The number of square degrees to load around the central latitude and longitude. 
         For reasonable loading times, set this as `0.1` or lower.
    """
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Update plotting functionality through rcParams
    mpl.rcParams.update({'figure.autolayout': True})
    
    # Define polygon bounds   
    latitude = (lat - buffer, lat + buffer)
    longitude = (lon - buffer, lon + buffer)

    # Define the bounding box that will be overlayed on the interactive map
    # The bounds are hard-coded to match those from the loaded data
    geom_obj = {
        "type": "Feature",
        "properties": {
            "style": {
                "stroke": True,
                "color": 'red',
                "weight": 4,
                "opacity": 0.8,
                "fill": True,
                "fillColor": False,
                "fillOpacity": 0,
                "showArea": True,
                "clickable": True
            }
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [
                        longitude[0],
                        latitude[0]
                    ],
                    [
                        longitude[1],
                        latitude[0]
                    ],
                    [
                        longitude[1],
                        latitude[1]
                    ],
                    [
                        longitude[0],
                        latitude[1]
                    ],
                    [
                        longitude[0],
                        latitude[0]
                    ]
                ]
            ]
        }
    }
    
    # Create a map geometry from the geom_obj dictionary
    # center specifies where the background map view should focus on
    # zoom specifies how zoomed in the background map should be
    loadeddata_geometry = ogr.CreateGeometryFromJson(str(geom_obj['geometry']))
    loadeddata_center = [
        loadeddata_geometry.Centroid().GetY(),
        loadeddata_geometry.Centroid().GetX()
    ]
    loadeddata_zoom = 16

    # define the study area map
    studyarea_map = Map(
        center=loadeddata_center,
        zoom=loadeddata_zoom,
        basemap=basemaps.Esri.WorldImagery
    )

    # define the drawing controls
    studyarea_drawctrl = DrawControl(
        polygon={"shapeOptions": {"fillOpacity": 0}},
        marker={},
        circle={},
        circlemarker={},
        polyline={},
    )

    # add drawing controls and data bound geometry to the map
    studyarea_map.add_control(studyarea_drawctrl)
    studyarea_map.add_layer(GeoJSON(data=geom_obj))

    # Index to count drawn polygons
    polygon_number = 0

    # Define widgets to interact with
    instruction = widgets.Output(layout={'border': '1px solid black'})
    with instruction:
        print("Draw a polygon within the red box to view a plot of "
              "average NDVI over time in that area.")

    info = widgets.Output(layout={'border': '1px solid black'})
    with info:
        print("Plot status:")

    fig_display = widgets.Output(layout=widgets.Layout(
        width="50%",  # proportion of horizontal space taken by plot
    ))

    with fig_display:
        plt.ioff()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_ylim([0, 1])

    colour_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Function to execute each time something is drawn on the map
    def handle_draw(self, action, geo_json):
        nonlocal polygon_number

        # Execute behaviour based on what the user draws
        if geo_json['geometry']['type'] == 'Polygon':

            info.clear_output(wait=True)  # wait=True reduces flicker effect
            
            # Save geojson polygon to io temporary file to be rasterized later
            jsonData = json.dumps(geo_json)
            binaryData = jsonData.encode()
            io = BytesIO(binaryData)
            io.seek(0)
            
            # Read the polygon as a geopandas dataframe
            gdf = gpd.read_file(io)
            gdf.crs = "EPSG:4326"

            # Convert the drawn geometry to pixel coordinates
            xr_poly = xr_rasterize(gdf, ds.NDVI.isel(time=0), crs='EPSG:6933')

            # Construct a mask to only select pixels within the drawn polygon
            masked_ds = ds.NDVI.where(xr_poly)
            
            masked_ds_mean = masked_ds.mean(dim=['x', 'y'], skipna=True)
            colour = colour_list[polygon_number % len(colour_list)]

            # Add a layer to the map to make the most recently drawn polygon
            # the same colour as the line on the plot
            studyarea_map.add_layer(
                GeoJSON(
                    data=geo_json,
                    style={
                        'color': colour,
                        'opacity': 1,
                        'weight': 4.5,
                        'fillOpacity': 0.0
                    }
                )
            )

            # add new data to the plot
            xr.plot.plot(
                masked_ds_mean,
                marker='*',
                color=colour,
                ax=ax
            )

            # reset titles back to custom
            ax.set_title("Average NDVI from Landsat 8")
            ax.set_xlabel("Date")
            ax.set_ylabel("NDVI")

            # refresh display
            fig_display.clear_output(wait=True)  # wait=True reduces flicker effect
            with fig_display:
                display(fig)
                
            with info:
                print("Plot status: polygon sucessfully added to plot.")

            # Iterate the polygon number before drawing another polygon
            polygon_number = polygon_number + 1

        else:
            info.clear_output(wait=True)
            with info:
                print("Plot status: this drawing tool is not currently "
                      "supported. Please use the polygon tool.")

    # call to say activate handle_draw function on draw
    studyarea_drawctrl.on_draw(handle_draw)

    with fig_display:
        # TODO: update with user friendly something
        display(widgets.HTML(""))

    # Construct UI:
    #  +-----------------------+
    #  | instruction           |
    #  +-----------+-----------+
    #  |  map      |  plot     |
    #  |           |           |
    #  +-----------+-----------+
    #  | info                  |
    #  +-----------------------+
    ui = widgets.VBox([instruction,
                       widgets.HBox([studyarea_map, fig_display]),
                       info])
    display(ui)
