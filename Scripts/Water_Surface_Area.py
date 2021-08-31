# Load modules, must include 'matplotlib'

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
from datacube.storage import masking

# Load utility functions
from deafrica_tools.datahandling import load_ard
from deafrica_tools.spatial import xr_rasterize
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.datahandling import wofs_fuser, mostcommon_crs
from deafrica_tools.plotting import display_map



def load_waterbody_data (lat, lon, buffer, start, end):
    
    """
    Loads WOFLs data for the study area within the time specified time period.
    Last modified: 26.08.2021
    
    Parameters 
    
    lat: float
        The central latitude to analyse
    lon: float
        The central longitude to analyse
    buffer:
         The number of square degrees to load around the central latitude and longitude
    start: 
        The start date for the observation period
    end: 
        The end date for the observation period
    Returns
    wofls: xarray.Dataset
        dataset containing 'wet' and 'dry' variables 
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Initialise the data cube. app used to identify this app
    dc = datacube.Datacube(app='Waterbody-app')
    
    # Define area to load
    longitude = (lon-buffer, lon+buffer)
    latitude = (lat+buffer, lat-buffer)
    
    # Specify the date range 
    time = (start, end)
    
    # Construct the data cube query 
    query = {
        'x': longitude,
        'y': latitude,
        'time' : time,
        'resolution' : (-30, 30)
    }
    
    # Load the data 
    output_crs = mostcommon_crs(dc=dc, product='ga_ls8c_wofs_2', query=query)
    wofls = dc.load(
        product = 'ga_ls8c_wofs_2',
        group_by="solar_day",
        fuse_func=wofs_fuser, 
        output_crs = output_crs,
        **query)

    # Dimension list for the dataset 
    dim_list = ['time', 'y', 'x']
    
    # Calculate the wet and dry pixels across all the pixels in an image 
    wofls['wet'] = masking.make_mask(wofls, wet=True).water
    wofls['dry'] = masking.make_mask(wofls, dry=True).water
    
    # Return the data 
    return(wofls)


def run_waterbody_app(ds, lat, lon, buffer, resolution=30):
    """
    Plots an interactive map of the water body study area and allows
    the user to draw polygons. This returns a plot of the Water Surface Area over time
    in the polygon area.
    Last modified: 25.08.2021
    
    Parameters
    ----------
    ds: xarray.Dataset 
        dataset containing 'wet' and 'dry' variables
    lat: float
        The central latitude corresponding to the area of loaded ds
    lon: float
        The central longitude corresponding to the area of loaded ds
    buffer:
        The number of square degrees to load around the central latitude and longitude. 
    resolution: 
        The size of each pixel in meters
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
    loadeddata_zoom = 11

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
              "Water Surface Area over time in that area.")

    info = widgets.Output(layout={'border': '1px solid black'})
    with info:
        print("Plot status:")

    fig_display = widgets.Output(layout=widgets.Layout(
        width="50%",  # proportion of horizontal space taken by plot
    ))

    with fig_display:
        plt.ioff()
        fig, ax = plt.subplots(figsize=(8, 6))


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
            xr_poly = xr_rasterize(gdf, ds.wet.isel(time=0), crs='EPSG:6933')

            # Construct a mask to only select pixels within the drawn polygon
            masked_ds = ds.wet.where(xr_poly)
            
            m_per_km = 1000
            area_per_pixel = resolution**2/m_per_km**2
            water_area = ds.wet.sum(dim=['x', 'y'])* area_per_pixel
            dry_area = ds.dry.sum(dim=['x', 'y'])* area_per_pixel
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
                water_area,
                marker='*',
                color=colour,
                ax=ax
            )
            
            # add aditional data to the plot, with different line symbology 
            xr.plot.plot(
                dry_area,
                marker='*',
                color=colour,
                linestyle = "--",
                ax=ax
            )

            # reset titles back to custom
            ax.set_title("Wet Area")
            ax.set_xlabel("Date")
            ax.set_ylabel("Water Surface Area")

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
    
    # Displays the user interface
    display(ui)
