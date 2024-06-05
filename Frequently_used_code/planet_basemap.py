import os
os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = f"{os.environ['JUPYTERHUB_SERVICE_PREFIX']}/proxy/{{port}}"

import warnings
import ipywidgets
from datetime import datetime, timedelta
from IPython.display import HTML, display
import numpy as np

from ipyleaflet import Map, LayersControl, TileLayer, basemap_to_tiles, basemaps, GeoData, FullScreenControl
from localtileserver import get_leaflet_tile_layer, TileClient


import datacube
from deafrica_tools.datahandling import load_ard
from deafrica_tools.plotting import rgb
from deafrica_tools.bandindices import calculate_indices
from datacube.utils.cog import write_cog


from deafrica_tools.waterbodies import (
    get_geohashes,
    get_waterbodies,
    get_waterbody,
    get_time_series,
    display_time_series,
)

# Turn off all warnings.
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")




def get_last_calendar_month():
    # Get the current date
    today = datetime.today()
    
    # Calculate the first day of the current month
    first_day_of_current_month = today.replace(day=1)
    # Subtract one day to get the last day of the previous month
    if today.day > 4:
        pre_month = 1
    else:
        pre_month = 45
    last_day_of_last_month = first_day_of_current_month - timedelta(days=pre_month)
    
    # Extract the year and month from the last day of the previous month
    year = last_day_of_last_month.year
    month = last_day_of_last_month.month
    
    return year, month

def loadplanet():
    dc = datacube.Datacube(app='planet')

    # Set the central latitude and longitude
    lat =  6.74248
    lon = -1.69340

    # Set the buffer to load around the central coordinates
    buffer = 0.05

    # Compute the bounding box coordinates
    xlim = (lon-buffer, lon+buffer)
    ylim =  (lat+buffer, lat-buffer)

    # Create a bounding box from study area coordinates
    bbox = (xlim[0], ylim[1], xlim[1], ylim[0])

    # Select all water bodies located within the bounding box
    polygons = get_waterbodies(bbox, crs="EPSG:4326")

    # load data
    ds = load_ard(dc=dc,
                  products=["s2_l2a"],
                  measurements=['red', 'green', 'blue', 'nir','swir_1'],
                  y=ylim,
                  x=xlim,
                  time=(f"2024"),
                  resolution=(-10, 10),
                  output_crs='EPSG:6933',
                  group_by="solar_day",
                  min_gooddata=1,
                  verbose=False,
                 )

    ds = ds.isel(time=-1)
    ds = calculate_indices(ds, index=['NDVI','BUI'], satellite_mission='s2')
    ds_ndvi = ds.where(ds.NDVI>=0.7, np.nan).NDVI
 
    write_cog(ds_ndvi, fname="ndvi.tif", overwrite=True)
    ds_bui = ds.where(ds.BUI>=0, np.nan).NDVI

    write_cog(ds_bui, fname="bui.tif", overwrite=True)
    year, month = get_last_calendar_month()
    planet_ = f"https://api.digitalearth.africa/planet/tiles/basemaps/v1/planet-tiles/planet_medres_visual_{year:04d}-{month:02d}_mosaic/gmap/"+"{z}/{x}/{y}.png"
    provider = TileLayer(url=planet_, name=f"Planet NICFI-{year:04d}-{month:02d}",  show_loading=True, attribution="Planet NICFI")
    provider.base = True

    geo_data = GeoData(geo_dataframe = polygons,
                       style={'color': 'black', 'fillColor': '#3366cc', 'opacity':0.05, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.6},
                       hover_style={'fillColor': 'red' , 'fillOpacity': 0.2},
                       name = 'Water Body')

    # First, create a tile server from local raster file
    bui_client = TileClient('bui.tif')
    ndvi_client = TileClient('ndvi.tif')
    # client = examples.get_elevation() # use example data

    # Create ipyleaflet tile layer from that server
    bui = get_leaflet_tile_layer(bui_client, nodata=np.nan, name='Built up area', colormap='Reds')
    ndvi = get_leaflet_tile_layer(ndvi_client, nodata=np.nan, name='Vegetation', colormap='Greens')

    openstreet = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    openstreet.base = True

    control = LayersControl(position='topright')
    # Create ipyleaflet map, add tile layer, and display
    m = Map(center=bui_client.center(), zoom=bui_client.default_zoom + 3)
    m.add(bui)
    m.add(ndvi)
    m.add(geo_data)
    m.add(openstreet)
    m.add(provider)

    m.add(FullScreenControl())
    m.add(control)

    return m