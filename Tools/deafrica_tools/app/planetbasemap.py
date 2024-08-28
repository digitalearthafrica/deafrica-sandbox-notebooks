"""
Functions for planet imagery and interacting with DE Africa dataset.
"""

import os

os.environ["LOCALTILESERVER_CLIENT_PREFIX"] = (
    f"{os.environ['JUPYTERHUB_SERVICE_PREFIX']}/proxy/{{port}}"
)

import warnings
from datetime import datetime

import datacube
import numpy as np
from datacube.utils.cog import write_cog
from ipyleaflet import (
    FullScreenControl,
    GeoData,
    LayersControl,
    LegendControl,
    Map,
    TileLayer,
    basemap_to_tiles,
    basemaps,
)
from localtileserver import TileClient, get_leaflet_tile_layer

from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.waterbodies import get_waterbodies

# Turn off all warnings.
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def get_last_calendar_month() -> tuple[int, int]:
    """
    Get the year and month of the previous calendar month.
    """
    # Get the current date
    today = datetime.today()

    current_year = today.year
    current_month = today.month

    # Extract the year and month of the previous month
    if current_month == 1:
        year = current_year - 1
        month = 12
    else:
        year = current_year
        month = current_month - 1

    return year, month


# Load map to display
def loadplanet(lon_range: tuple, lat_range: tuple, threshold_nvdi: float, threshold_bui: float):
    """
    Loading of planet imagery and S2 R
    """
    # Connect to the datacube
    dc = datacube.Datacube(app="planet")

    # Create a bounding box from study area coordinates
    bbox = (lon_range[0], lat_range[0], lon_range[1], lat_range[1])

    # Select all water bodies located within the bounding box
    polygons = get_waterbodies(bbox, crs="EPSG:4326")

    # Get the year and month of the previous calendar month.
    year, month = get_last_calendar_month()

    # Load the S2 Annual GeoMAD
    ds = dc.load(
        product="gm_s2_annual",
        measurements=["red", "green", "blue", "nir", "swir_1"],
        y=lat_range,
        x=lon_range,
        resolution=(-10, 10),
        time=(f"{year:04d}"),
    )
    # Check if there was any current data in the previous load
    if len(ds) == 0:
        # if  not then load with the manual date provider
        ds = dc.load(
            product="gm_s2_annual",
            measurements=["red", "green", "blue", "nir", "swir_1"],
            y=lat_range,
            x=lon_range,
            resolution=(-10, 10),
            time=("2022", f"{year:04d}"),
        )

    # select the last best image
    ds = ds.isel(time=-1)

    # calculate NDVI and BUI
    ds = calculate_indices(ds, index=["NDVI", "BUI"], satellite_mission="s2")

    # Select NDVI value greater than threshold_nvdi value
    ds_ndvi = ds.where(ds.NDVI >= threshold_nvdi, np.nan).NDVI

    # Save the ndvi raster file
    write_cog(ds_ndvi, fname="ndvi.tif", overwrite=True)

    # Select BUI greater than 0
    ds_bui = ds.where(ds.BUI >= threshold_bui, np.nan).BUI

    # Save the bui raster file
    write_cog(ds_bui, fname="bui.tif", overwrite=True)

    # Defining the planet api
    planet_ = (
        f"https://api.digitalearth.africa/planet/tiles/basemaps/v1/planet-tiles/planet_medres_visual_{year:04d}-{month:02d}_mosaic/gmap/"
        + "{z}/{x}/{y}.png"
    )

    # Converting the api to tilelayer
    planet_tile = TileLayer(
        url=planet_,
        name=f"Planet NICFI-{year:04d}-{month:02d}",
        show_loading=True,
        attribution="Planet NICFI",
    )
    planet_tile.base = True

    # Converting the water polygon to GeoData
    geo_data = GeoData(
        geo_dataframe=polygons,
        style={
            "color": "black",
            "fillColor": "#3366cc",
            "opacity": 0.05,
            "weight": 1.9,
            "dashArray": "2",
            "fillOpacity": 0.6,
        },
        name="Water Body",
    )

    # Create a tile server from local raster file
    bui_client = TileClient("bui.tif")
    ndvi_client = TileClient("ndvi.tif")

    # Create ipyleaflet tile layer from that server
    tile_info = f"<a href=https://explorer.digitalearth.africa/products/gm_s2_annual> DE Africa Sentinel-2 Annual GeoMAD {ds.time.dt.year.values} </a>"
    bui = get_leaflet_tile_layer(
        bui_client,
        nodata=np.nan,
        name="Built up area",
        colormap="Reds",
        vmin=-1,
        vmax=1,
        attribution=tile_info,
    )
    ndvi = get_leaflet_tile_layer(
        ndvi_client, nodata=np.nan, name="Vegetation", colormap="Greens", vmin=0, vmax=1
    )

    # Openstreet basemap
    openstreet = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    openstreet.base = True

    # Create ipyleaflet map, add tile layer, and display
    map_con = Map(center=bui_client.center(), zoom=bui_client.default_zoom + 2)

    control = LayersControl(position="topright", collapsed=False)

    legend = LegendControl(
        {
            "Built-up Area": "#ff8533",
            "\n": "#fff",
            "Vegetation": "#34a832",
            " ": "#fff",
            "Water body": "#1d18c4",
        },
        title="Legend",
        position="bottomleft",
    )

    # add the created layer and control to the map
    map_con.add(bui)
    map_con.add(ndvi)
    map_con.add(geo_data)
    map_con.add(openstreet)
    map_con.add(planet_tile)
    map_con.add(FullScreenControl())
    map_con.add(control)
    map_con.add(legend)

    return map_con
