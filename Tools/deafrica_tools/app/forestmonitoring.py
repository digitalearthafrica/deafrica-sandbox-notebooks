'''
Functions for loading and interacting with Global Forest Change data in the forest monitoring notebook, inside the Real_world_examples folder.
'''

# Import required packages

# Force GeoPandas to use Shapely instead of PyGEOS
# In a future release, GeoPandas will switch to using Shapely by default.
import os
os.environ['USE_PYGEOS'] = '0'

import json
import warnings
from io import BytesIO

import deafrica_tools.app.widgetconstructors as deawidgets
import geopandas as gpd
import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from deafrica_tools.dask import create_local_dask_cluster
from deafrica_tools.spatial import xr_rasterize
from ipyleaflet import (
    DrawControl,
    GeoData,
    LayerGroup,
    LayersControl,
    Map,
    WidgetControl,
    WMSLayer,
    basemap_to_tiles,
    basemaps,
)
from ipywidgets import HTML, Button, GridspecLayout, HBox, Layout, Output, VBox
from matplotlib.patches import Patch
from traitlets import Unicode

# Turn off all warnings.
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

def make_box_layout():
    """
    Defines a number of CSS properties that impact how a widget is laid out.
    """
    return Layout(  # border='solid 1px black',
        margin="0px 10px 10px 0px",
        padding="5px 5px 5px 5px",
        width="100%",
        height="100%",
    )

def create_expanded_button(description, button_style):
    """
    Defines a number of CSS properties to create a button to handle mouse clicks.
    """
    return Button(
        description=description,
        button_style=button_style,
        layout=Layout(width="auto", height="auto"),
    )

def load_gfclayer(gdf_drawn, gfclayer):
    """
    Loads the selected Global Forest Change layer for the
    area drawn on the map widget.
    """
    # Configure local dask cluster.
    client = create_local_dask_cluster(return_client=True, display_client=True)

    # Get the coordinates of the top-left corner for each Global Forest Change tile,
    # covering the area of interest.
    min_lat, max_lat = (
        gdf_drawn.bounds.miny.item(),
        gdf_drawn.bounds.maxy.item(),
    )
    min_lon, max_lon = (
        gdf_drawn.bounds.minx.item(),
        gdf_drawn.bounds.maxx.item(),
    )

    lats = np.arange(
        np.floor(min_lat / 10) * 10, np.ceil(max_lat / 10) * 10, 10
    ).astype(int)
    lons = np.arange(
        np.floor(min_lon / 10) * 10, np.ceil(max_lon / 10) * 10, 10
    ).astype(int)

    coord_list = []
    for lat in lats:
        lat = lat + 10
        if lat >= 0:
            lat_str = f"{lat:02d}N"
        else:
            lat_str = f"{abs(lat):02d}S"
        for lon in lons:
            if lon >= 0:
                lon_str = f"{lon:03d}E"
            else:
                lon_str = f"{abs(lon):03d}W"
            coord_str = f"{lat_str}_{lon_str}"
            coord_list.append(coord_str)

    # Load each Global Forest Change tile covering the area of interest.
    base_url = f"https://storage.googleapis.com/earthenginepartners-hansen/GFC-2021-v1.9/Hansen_GFC-2021-v1.9_{gfclayer}_"
    dask_chunks = dict(x=2048, y=2048)

    tile_list = []
    for coord in coord_list:
        tile_url = f"{base_url}{coord}.tif"
        # Load the tile as an xarray.DataArray.
        tile = rioxarray.open_rasterio(tile_url, chunks=dask_chunks).squeeze()
        tile_list.append(tile)

    # Merge the tiles into a single xarray.DataArray.
    ds = xr.combine_by_coords(tile_list)
    # Clip the dataset using the bounds of the area of interest.
    ds = ds.rio.clip_box(
        minx=min_lon - 0.00025,
        miny=min_lat - 0.00025,
        maxx=max_lon + 0.00025,
        maxy=max_lat + 0.00025,
    )
    # Rename the y and x variables for DEA convention on xarray.DataArrays where crs="EPSG:4326".
    ds = ds.rename({"y": "latitude", "x": "longitude"})

    # Mask pixels representing no loss (encoded as 0) in the "lossyear" layer.
    if gfclayer == "lossyear":
        ds = ds.where(ds != 0)
    # Mask pixels representing no gain (encoded as 0) in the "gain" layer.
    elif gfclayer == "gain":
        ds = ds.where(ds != 0)
    # Mask pixels with 0 percentage tree canopy cover.
    elif gfclayer == "treecover2000":
        ds = ds.where(ds != 0)

    # Create a mask from the area of interest GeoDataFrame.
    mask = xr_rasterize(gdf_drawn, ds)
    # Mask the dataset.
    ds = ds.where(mask)
    # Convert the xarray.DataArray to a dataset.
    ds = ds.to_dataset(name=gfclayer)
    # Compute.
    ds = ds.compute()
    # Assign the "EPSG:4326" CRS to the dataset.
    ds.rio.write_crs(4326, inplace=True)
    ds = ds.transpose("latitude", "longitude")

    # Close down the dask client.
    client.close()
    return ds

def load_all_gfclayers(gdf_drawn):
    gfclayers = ["treecover2000", "gain", "lossyear"]

    dataset_list = []
    for layer in gfclayers:
        ds = load_gfclayer(gdf_drawn, gfclayer=layer)
        dataset_list.append(ds)

    dataset = xr.merge(dataset_list)
    return dataset

def get_gfclayer_treecover2000(gfclayer_ds, gfclayer="treecover2000"):
    """
    Preprocess the Global Forest change "treecover2020" layer.
    """
    ds = gfclayer_ds[gfclayer]

    # Check if the dataarray is empty.
    condition = ds.isnull().all().item()

    if condition:
        return None
    else:
        # Mask the dataset.
        mask = np.isnan(ds)
        ds_masked = ds.where(mask, 1)

        # Get the pixel count for each unique pixel value in the layer.
        counts = np.unique(ds_masked, return_counts=True)
        # Remove the counts for pixels with the value np.nan.
        index = np.argwhere(np.isnan(counts[0]))
        counts_dict = dict(
            zip(np.delete(counts[0], index), np.delete(counts[1], index))
        )

        # Reproject the dataset to EPSG:6933 which uses metres
        ds_reprojected = ds_masked.rio.reproject("EPSG:6933")
        # Get the area per pixel.
        pixel_length = ds_reprojected.geobox.resolution[1]
        m_per_km = 1000
        per_pixel_area = (pixel_length / m_per_km) ** 2

        # Save the results as a pandas DataFrame.
        df = pd.DataFrame(
            data={
                "Year": ["2000"],
                "Tree Cover in km$^2$": np.fromiter(counts_dict.values(), dtype=float)
                * per_pixel_area,
            }
        )

        # Get the total area.
        print_statement = f'Total Forest Cover in {df["Year"].item()}: {round(df["Tree Cover in km$^2$"].item(), 4)} km2'

        # File name to use when exporting results.
        file_name = f"forest_cover_in_2000"

        return ds, df, print_statement, file_name

def get_gfclayer_gain(gfclayer_ds, gfclayer="gain"):
    """
    Preprocess the Global Forest Change "gain" layer.
    """
    ds = gfclayer_ds[gfclayer]

    # Check if the dataarray is empty.
    condition = ds.isnull().all().item()

    if condition:
        return None
    else:
        # Get the pixel count for each unique pixel value in the layer.
        counts = np.unique(ds, return_counts=True)
        # Remove the counts for pixels with the value np.nan.
        index = np.argwhere(np.isnan(counts[0]))
        counts_dict = dict(
            zip(np.delete(counts[0], index), np.delete(counts[1], index))
        )

        # Reproject the dataset to EPSG:6933 which uses metres.
        ds_reprojected = ds.rio.reproject("EPSG:6933")
        # Get the area per pixel.
        pixel_length = ds_reprojected.geobox.resolution[1]
        m_per_km = 1000
        per_pixel_area = (pixel_length / m_per_km) ** 2

        # Save the results as a pandas DataFrame.
        df = pd.DataFrame(
            data={
                "Year": ["2000-2012"],
                "Forest Cover Gain in km$^2$": np.fromiter(
                    counts_dict.values(), dtype=float
                )
                * per_pixel_area,
            }
        )

        # Get the total area.
        print_statement = f'Total Forest Cover Gain {df["Year"].item()}: {round(df["Forest Cover Gain in km$^2$"].item(), 4)} km2'

        # File name to use when exporting results.
        file_name = f"forest_cover_gain_from_2000_to_2012"

        return ds, df, print_statement, file_name

def get_gfclayer_lossyear(gfclayer_ds, start_year, end_year, gfclayer="lossyear"):
    """
    Preprocess the Global Forest Change "lossyear" layer.
    """

    ds = gfclayer_ds[gfclayer]

    # Mask the dataset to the selected time range.
    selected_years = list(range(start_year, end_year + 1))
    mask = ds.isin(selected_years)
    ds = ds.where(mask)

    # Check if the dataarray is empty.
    condition = ds.isnull().all().item()

    if condition:
        return None
    else:
        # Get the pixel count for each unique pixel value in the layer.
        counts = np.unique(ds, return_counts=True)
        # Remove the counts for pixels with the value np.nan.
        index = np.argwhere(np.isnan(counts[0]))
        counts_dict = dict(
            zip(np.delete(counts[0], index), np.delete(counts[1], index))
        )

        # Reproject the dataset to EPSG:6933 which uses metres
        ds_reprojected = ds.rio.reproject("EPSG:6933")
        # Get the area per pixel.
        pixel_length = ds_reprojected.geobox.resolution[1]
        m_per_km = 1000
        per_pixel_area = (pixel_length / m_per_km) ** 2

        # For each year get the area of loss.
        # Save the results as a pandas DataFrame.
        df = pd.DataFrame(
            {
                "Year": 2000 + np.fromiter(counts_dict.keys(), dtype=int),
                "Forest Cover Loss in km$^2$": np.fromiter(
                    counts_dict.values(), dtype=float
                )
                * per_pixel_area,
            }
        )

        # Get the total area.
        print_statement = f'Total Forest Cover Loss from {start_year + 2000} to {end_year + 2000}: {round(df["Forest Cover Loss in km$^2$"].sum(), 4)} km2'

        # File name to use when exporting results.
        file_name = f"forest_cover_loss_from_{start_year + 2000}_to_{end_year + 2000}"

        return ds, df, print_statement, file_name

def plot_gfclayer_treecover2000(gfclayer_ds, gfclayer="treecover2000"):
    """
    Plot the Global Forest Change "treecover2000" layer.
    """

    if get_gfclayer_treecover2000(gfclayer_ds) is None:
        print(
            f"No Global Forest Change {gfclayer} layer data found in the selected area. Please select a new polygon over an area with data."
        )
    else:
        ds, df, print_statement, file_name = get_gfclayer_treecover2000(gfclayer_ds)

        # Export the dataframe as a csv.
        df.to_csv(f"{file_name}.csv", index=False)
        print(f'Table exported to "{file_name}.csv"')

        # Define the plotting parameters.
        figure_width = 10
        figure_length = 10
        title = f"Tree Canopy Cover for the Year 2000"

        # Plot the dataset.
        fig, ax = plt.subplots(figsize=(figure_width, figure_length))
        im = ds.plot(cmap="Greens", add_colorbar=False, ax=ax)
        # Add a colorbar to the plot.
        cbar = plt.colorbar(mappable=im)
        cbar.set_label(
            "Percentage tree canopy cover for year 2000", labelpad=-65, y=0.25
        )
        # Add a title to the plot.
        plt.title(title)
        # Save the plot.
        plt.savefig(f"{file_name}.png")
        print(f'Figure exported to "{file_name}.png"')
        plt.show()

        print(print_statement)

def plot_gfclayer_gain(gfclayer_ds, gfclayer="gain"):
    """
    Plot the Global Forest Change "gain" layer.
    """

    if get_gfclayer_gain(gfclayer_ds) is None:
        print(
            f"No Global Forest Change {gfclayer} layer data found in the selected area. Please select a new polygon over an area with data."
        )
    else:
        ds, df, print_statement, file_name = get_gfclayer_gain(gfclayer_ds)

        # Export the dataframe as a csv.
        df.to_csv(f"{file_name}.csv", index=False)
        print(f'Table exported to "{file_name}.csv"')

        # Define the plotting parameters.
        color = "#6CAE75"
        figure_width = 10
        figure_length = 10
        title = f"Forest Cover Gain from 2000 to 2012"

        # Plot the dataset.
        fig, ax = plt.subplots(figsize=(figure_width, figure_length))
        im = ds.plot(cmap=mcolors.ListedColormap([color]), add_colorbar=False, ax=ax)
        # Add a legend to the plot.
        im.axes.legend(
            [Patch(facecolor=color)],
            ["Global forest cover gain 2000–2012"],
            loc="lower left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )
        # Add a title to the plot.
        plt.title(title)
        # Save the plot.
        plt.savefig(f"{file_name}.png")
        print(f'Figure exported to "{file_name}.png"')
        plt.show()

        print(print_statement)

def plot_gfclayer_lossyear(gfclayer_ds, start_year, end_year, gfclayer="lossyear"):
    """
    Plot the Global Forest change "lossyear" layer.
    """

    if (
        get_gfclayer_lossyear(gfclayer_ds, start_year, end_year, gfclayer="lossyear")
        is None
    ):
        print(
            f"No Global Forest Change {gfclayer} layer data found in the selected area. Please select a new polygon over an area with data."
        )
    else:
        ds, df, print_statement, file_name = get_gfclayer_lossyear(
            gfclayer_ds, start_year, end_year, gfclayer="lossyear"
        )

        # Export the dataframe as a csv.
        df.to_csv(f"{file_name}.csv", index=False)
        print(f'Table exported to "{file_name}.csv"')

        # Define the plotting parameters.
        figure_width = 10
        figure_length = 15
        nrows = 2
        ncols = 1
        title = f"Forest Cover Loss from {start_year + 2000} to {end_year + 2000}"

        # Location of transition from one color to the next on the colormap.
        color_levels = list(np.arange(1 - 0.5, 22, 1))
        # Ticks to be displayed.
        ticks = list(np.arange(1, 22))
        tick_labels = list(2000 + np.arange(1, 22))

        # Define the color map to use when plotting.
        color_list = [
            "#e6194b",
            "#3cb44b",
            "#ffe119",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#46f0f0",
            "#f032e6",
            "#bcf60c",
            "#fabebe",
            "#008080",
            "#e6beff",
            "#9a6324",
            "#fffac8",
            "#800000",
            "#aaffc3",
            "#808000",
            "#ffd8b1",
            "#000075",
            "#808080",
            "#7A306C",
        ]
        cmap = mcolors.ListedColormap(colors=color_list, N=21)
        norm = mcolors.BoundaryNorm(boundaries=color_levels, ncolors=cmap.N)

        # Plot the dataset.
        fig, (ax1, ax2) = plt.subplots(
            nrows, ncols, figsize=(figure_width, figure_length)
        )
        im = ds.plot(ax=ax1, cmap=cmap, norm=norm, add_colorbar=False)
        # Add a title to the subplot.
        ax1.set_title(title)
        # Add a colorbar to the subplot.
        cbar = plt.colorbar(mappable=im, ticks=ticks)
        cbar.set_label("Year of gross forest cover loss event", labelpad=-60, y=0.25)
        cbar.set_ticklabels(tick_labels)
        # Plot the second subplot.
        df.plot(
            x="Year",
            y="Forest Cover Loss in km$^2$",
            ylabel="Forest Cover Loss in km$^2$",
            title=title,
            ax=ax2,
        )
        # Save the plot.
        plt.savefig(f"{file_name}.png")
        print(f'Figure exported to "{file_name}.png"')
        plt.show()

        print(print_statement)

def plot_gfclayer_all(gfclayer_ds, start_year, end_year):
    """
    Plot all the Global Forest Change Layers loaded.
    """

    # Define the plotting parameters.
    figure_width = 10
    figure_length = 10
    treecover_color = "Greens"
    gain_color = "yellow"
    lossyear_color = "red"

    print_statement_list = []
    filename_list = ["\nTables exported as: "]

    figure_fn = "global_forest_change_all_layers.png"

    # Define the figure.
    fig, ax = plt.subplots(figsize=(figure_width, figure_length))
    if (
        get_gfclayer_treecover2000(
            gfclayer_ds[["treecover2000"]], gfclayer="treecover2000"
        )
        is None
    ):
        print(
            f"No Global Forest Change 'treecover2000' layer data found in the selected area. Please select a new polygon over an area with data."
        )
    else:
        (
            ds_treecover2000,
            df_treecover2000,
            print_statement_treecover2000,
            file_name_treecover2000,
        ) = get_gfclayer_treecover2000(
            gfclayer_ds[["treecover2000"]], gfclayer="treecover2000"
        )
        # Plot the treecover2000 layer as the background layer.
        background = ds_treecover2000.plot(
            cmap=treecover_color, add_colorbar=False, ax=ax
        )
        # Add a colorbar to the treecover2000 plot.
        cbar = plt.colorbar(mappable=background)
        cbar.set_label(
            "Percentage tree canopy cover for year 2000", labelpad=-65, y=0.25
        )
        # Export the dataframe as a csv.
        df_treecover2000.to_csv(f"{file_name_treecover2000}.csv", index=False)
        # Add the print statement to the list.
        print_statement_list.append(print_statement_treecover2000)
        # Add the file name to the list.
        filename_list.append(f'"{file_name_treecover2000}.csv"')

    if get_gfclayer_gain(gfclayer_ds[["gain"]], gfclayer="gain") is None:
        print(
            f"No Global Forest Change 'gain' layer data found in the selected area. Please select a new polygon over an area with data."
        )
    else:
        ds_gain, df_gain, print_statement_gain, file_name_gain = get_gfclayer_gain(
            gfclayer_ds[["gain"]], gfclayer="gain"
        )
        # Plot the gain layer.
        ds_gain.plot(
            ax=ax, cmap=mcolors.ListedColormap([gain_color]), add_colorbar=False
        )
        # Export the dataframe as a csv.
        df_gain.to_csv(f"{file_name_gain}.csv", index=False)
        # Add the print statement to the list.
        print_statement_list.append(print_statement_gain)
        # Add the file name to the list.
        filename_list.append(f'"{file_name_gain}.csv"')

    if (
        get_gfclayer_lossyear(
            gfclayer_ds[["lossyear"]], start_year, end_year, gfclayer="lossyear"
        )
        is None
    ):
        print(
            f"No Global Forest Change 'lossyear' layer data found in the selected area. Please select a new polygon over an area with data."
        )
    else:
        (
            ds_lossyear,
            df_lossyear,
            print_statement_lossyear,
            file_name_lossyear,
        ) = get_gfclayer_lossyear(
            gfclayer_ds[["lossyear"]], start_year, end_year, gfclayer="lossyear"
        )
        # Plot the lossyear layer.
        ds_lossyear.plot(
            ax=ax, cmap=mcolors.ListedColormap([lossyear_color]), add_colorbar=False
        )
        # Export the dataframe as a csv.
        df_lossyear.to_csv(f"{file_name_lossyear}.csv", index=False)
        # Add the print statement to the list.
        print_statement_list.append(print_statement_lossyear)
        # Add the file name to the list.
        filename_list.append(f'"{file_name_lossyear}.csv"')

    # Add a legend to the plot.
    ax.legend(
        [Patch(facecolor=gain_color), Patch(facecolor=lossyear_color)],
        [
            "Global forest cover \n gain 2000–2012",
            f"Global forest cover \n loss {str(2000+start_year)}-{str(2000+end_year)}",
        ],
        loc="lower right",
        bbox_to_anchor=(-0.1, 0.75),
        frameon=False,
    )

    plt.title("Global Forest Change Layers")
    plt.savefig(figure_fn)
    plt.show()
    print(*print_statement_list, sep="\n")
    print(*filename_list, sep="\n\t")
    print(f'\nFigure saved as "{figure_fn}"');

def plot_gfclayer(gfclayer_ds, start_year, end_year, gfclayer):
    if gfclayer == "treecover2000":
        plot_gfclayer_treecover2000(gfclayer_ds, gfclayer)
    elif gfclayer == "lossyear":
        plot_gfclayer_lossyear(gfclayer_ds, start_year, end_year, gfclayer)
    elif gfclayer == "gain":
        plot_gfclayer_gain(gfclayer_ds, gfclayer)
    elif gfclayer == "alllayers":
        plot_gfclayer_all(gfclayer_ds, start_year, end_year)

def update_map_layers(self):
    """
    Updates map widget to add new basemap when selected
    using menu options.
    """
    # Clear data load parameters to trigger data reload.
    self.gfclayer_ds = None

    # Remove all layers from the map_layers Layers Group.
    self.map_layers.clear_layers()
    # Add the selected basemap to the layer Group.
    self.map_layers.add_layer(self.basemap)

class forest_monitoring_app(HBox):
    def __init__(self):
        super().__init__()

        ##################
        # HEADER FOR APP #
        ##################

        # Create the header widget.
        header_title_text = "<h3>Digital Earth Africa Forest Change</h3>"
        instruction_text = """<p>Select the desired Global Forest Change layer, then zoom in and draw a polygon to
                                select an area for which to plot the selected Global Forest Change layer. Alternatively, <b>upload a vector file</b> of the area of interest.</p>"""
        self.header = deawidgets.create_html(
            value=f"{header_title_text}{instruction_text}"
        )
        self.header.layout = make_box_layout()

        ############################
        # WIDGETS FOR APP CONTROLS #
        ############################

        ## Selection widget for selecting the basemap to use for the map widget.
        ## and when plotting the Global Forest Change Layer.
        # Basemaps available for selection for the map widget.
        self.basemap_list = [
            ("Open Street Map", basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)),
            ("ESRI World Imagery", basemap_to_tiles(basemaps.Esri.WorldImagery)),
        ]
        # Set the default basemap to be used for the map widget / initial value for the widget.
        self.basemap = self.basemap_list[0][1]
        # Dropdown selection widget.
        dropdown_basemap = deawidgets.create_dropdown(
            options=self.basemap_list, value=self.basemap
        )
        # Register the update function to run when a new value is selected
        # on the dropdown_basemap widget.
        dropdown_basemap.observe(self.update_basemap, "value")
        # Text to accompany the dropdown selection widget.
        basemap_selection_html = deawidgets.create_html(
            value=f"</br><b>Map overlay:</b>"
        )
        # Combine the basemap_selection_html text and the dropdown_basemap widget in a single container.
        basemap_selection = VBox([basemap_selection_html, dropdown_basemap])

        ## Selection widget for selecting the Global Forest change layer to plot.
        # Global Forest Change layers available plotting.
        self.gfclayers_list = [
            ("Year of gross forest cover loss event", "lossyear"),
            ("Global forest cover gain 2000–2012", "gain"),
            ("Tree canopy cover for the year 2000", "treecover2000"),
            ("All layers", "alllayers"),
        ]
        # Set the default GFC layer to be plotted / initial value for the widget.
        self.gfclayer = self.gfclayers_list[0][1]

        ## Selection widget for the data time range.
        # Set the default time range for which to load data for.
        self.start_year = 1
        self.end_year = 21

        # Create the time range selector.
        time_range = list(range(self.start_year, self.end_year + 1))
        time_range_str = [str(2000 + i) for i in time_range]

        timerange_options = tuple(zip(time_range_str, time_range))
        timerange_selection_slide = widgets.SelectionRangeSlider(
            options=timerange_options,
            value=(self.start_year, self.end_year),
            description="",
            disabled=False,
        )
        # Register the update function to run when a new value is selected on the slider.
        timerange_selection_slide.observe(self.update_timerange, "value")
        # Text to accompany the timerange_selection widget.
        timerange_selection_html = deawidgets.create_html(
            value=f"</br><b>Forest Cover Loss Time Range:</b>"
        )
        # Combine the timerange_selection_text and the timerange_selection_slide  in a single container.
        timerange_selection = VBox(
            [timerange_selection_html, timerange_selection_slide]
        )

        # Set the initial parameter for the GFC layer dataset.
        self.gfclayer_ds = None
        # Dropdown selection widget.
        dropdown_gfclayer = deawidgets.create_dropdown(
            options=self.gfclayers_list, value=self.gfclayer
        )
        # Register the update function to run when a new value is selected
        # on the dropdown_gfclayer widget.
        dropdown_gfclayer.observe(self.update_gfclayer, "value")
        # Text to accompany the dropdown selection widget.
        gfclayer_selection_html = deawidgets.create_html(
            value=f"</br><b>Global Forest Change Layer:</b>"
        )
        # Combine the gfclayer_selection_html text and the dropdown_gfclayer widget in a single container.
        gfclayer_selection = VBox([gfclayer_selection_html, dropdown_gfclayer])

        ## Add a checkbox for whether to overide the limit to the size of polygon drawn on the
        ## map widget.
        # Initial value of the widget.
        self.max_size = False
        # CheckBox widget.
        checkbox_max_size = deawidgets.create_checkbox(
            value=self.max_size, description="Enable", layout={"width": "95%"}
        )
        # Text to accompany the CheckBox widget.
        checkbox_max_size_html = deawidgets.create_html(
            value=f"""</br><b>Override maximum size limit: 
                                                            (use with caution; may cause memory issues/crashes)<b>"""
        )
        # Register the update function to run when the checkbox is ticked.
        # on the checkbox_max_size CheckBox
        checkbox_max_size.observe(self.update_checkbox_max_size, "value")
        # # Combine the checkbox_max_size_html text and the checkbox_max_size widget in a single container.
        enable_max_size = VBox([checkbox_max_size_html, checkbox_max_size])
        
        # Add widget to enable uploading a geojson or ESRI shapefile. 
        self.gdf_uploaded = None
        fileupload_aoi = widgets.FileUpload(accept="", multiple=True)
        # Register the update function to be called for the file upload. 
        fileupload_aoi.observe(self.update_fileupload_aoi, "value")
        fileupload_html = deawidgets.create_html(value=f"""</br><i><b>Advanced</b></br>Upload a GeoJSON or ESRI Shapefile (<5 mb) containing a single area of interest.</i>""")
        fileupload = VBox([fileupload_html, fileupload_aoi])
        
        
        ## Put the app controls widgets into a single container.
        parameter_selection = VBox(
            [
                basemap_selection,
                gfclayer_selection,
                timerange_selection,
                enable_max_size,
                fileupload
            ]
        )
        parameter_selection.layout = make_box_layout()

        ## Button to click to run the app.
        run_button = create_expanded_button(
            description="Generate plot", button_style="info"
        )
        # Register the update function to be called when the run_button button
        # is clicked.
        run_button.on_click(self.run_app)
        
        

        ###########################
        # WIDGETS FOR APP OUTPUTS #
        ###########################

        self.status_info = Output(layout=make_box_layout())
        self.output_plot = Output(layout=make_box_layout())

        #################################
        # MAP WIDGET WITH DRAWING TOOLS #
        #################################

        # Create the map widget.
        self.m = deawidgets.create_map(
            map_center=(-18.45, 28.93),
            zoom_level=11,
        )
        self.m.layout = make_box_layout()

        # Create an empty Layer Group.
        self.map_layers = LayerGroup(layers=())
        # Name of the Layer Group layer.
        self.map_layers.name = "Map Overlays"
        # Add the empty Layer Group as a single layer to the map widget.
        self.m.add_layer(self.map_layers)

        # Create the desired drawing tools.
        desired_drawtools = ["rectangle", "polygon"]
        draw_control = deawidgets.create_drawcontrol(desired_drawtools)
        # Add drawing tools to the map widget.
        self.m.add_control(draw_control)
        # Set the initial parameters for the drawing tools.
        self.target = None
        self.action = None
        self.gdf_drawn = None

        #####################################
        # HANDLER FUNCTION FOR DRAW CONTROL #
        #####################################

        def handle_draw(target, action, geo_json):

            """
            Defines the action to take once something is drawn on the
            map widget.
            """
            # Remove previously uploaded data if present
            self.gdf_uploaded = None
            fileupload_aoi._counter = 0
            
            self.target = target
            self.action = action

            # Clear data load parameters to trigger data reload.
            self.gfclayer_ds = None

            # Convert the drawn polygon geojson to a GeoDataFrame.
            json_data = json.dumps(geo_json)
            binary_data = json_data.encode()
            io = BytesIO(binary_data)
            io.seek(0)
            gdf = gpd.read_file(io)
            gdf.crs = "EPSG:4326"

            # Convert the GeoDataFrame to WGS 84 / NSIDC EASE-Grid 2.0 Global and compute the area.
            gdf_drawn_nsidc = gdf.copy().to_crs("EPSG:6933")
            m2_per_ha = 10000
            area = gdf_drawn_nsidc.area.values[0] / m2_per_ha

            polyarea_label = (
                f"Total area of Global Forest Change {self.gfclayer} layer to load"
            )
            polyarea_text = f"<b>{polyarea_label}</b>: {area:.2f} ha</sup>"

            # Test the size of the polygon drawn.
            if self.max_size:
                confirmation_text = """<span style="color: #33cc33">  
                                    <b>(Overriding maximum size limit; use with caution as may lead to memory issues)</b></span>"""
                self.header.value = (
                    header_title_text
                    + instruction_text
                    + polyarea_text
                    + confirmation_text
                )
                self.gdf_drawn = gdf
            elif area <= 50000:
                confirmation_text = """<span style="color: #33cc33">
                                    <b>(Area to extract falls within
                                    recommended 50000 ha limit)</b></span>"""
                self.header.value = (
                    header_title_text
                    + instruction_text
                    + polyarea_text
                    + confirmation_text
                )
                self.gdf_drawn = gdf
            else:
                warning_text = """<span style="color: #ff5050">
                                <b>(Area to extract is too large,
                                please select an area less than 50000 )</b></span>"""
                self.header.value = (
                    header_title_text + instruction_text + polyarea_text + warning_text
                )
                self.gdf_drawn = None

        # Register the handler for draw events.
        draw_control.on_draw(handle_draw)

        ###############################
        # SPECIFICATION OF APP LAYOUT #
        ###############################

        # Create the app layout.
        grid_rows = 12
        grid_columns = 11
        grid_height = "1500px"
        grid_width = "auto"
        grid = GridspecLayout(
            grid_rows, grid_columns, height=grid_height, width=grid_width
        )

        # Place app widgets and components in app layout.
        # [rows, columns]
        grid[0, :] = self.header
        grid[1:6, 0:4] = parameter_selection
        grid[6, 0:4] = run_button
        grid[7:, 0:4] = self.status_info
        grid[6:, 4:] = self.output_plot
        grid[1:6, 4:] = self.m
        # Display using HBox children attribute
        self.children = [grid]

    ######################################
    # DEFINITION OF ALL UPDATE FUNCTIONS #
    ######################################

    def update_basemap(self, change):
        """
        Updates the basemap on the map widget based on the
        selected value of the dropdown_basemap widget.
        """
        self.basemap = change.new
        self.output_plot_basemap = get_basemap(self.basemap.url)
        update_map_layers(self)

    def update_gfclayer(self, change):
        """
        Updates the Global Forest Change layer to be plotted
        based on the selected value of the dropdown_gfclayer widget.
        """
        self.gfclayer = change.new

    def update_timerange(self, change):
        """Updates the time range of the data to be loaded"""
        self.start_year = change.new[0]
        self.end_year = change.new[1]

    def update_checkbox_max_size(self, change):
        """
        Sets the value of self.max_size to True when the
        checkbox_max_size CheckBox is checked.
        """
        self.max_size = change.new
        
    def update_fileupload_aoi(self, change):

        # Clear any drawn data if present
        self.gdf_drawn = None
    
        # Save to file
        for uploaded_filename in change.new.keys():
            with open(uploaded_filename, "wb") as output_file:
                content = change.new[uploaded_filename]['content']
                output_file.write(content)

        with self.status_info:

            try:            

                print('Loading vector data...', end='\r')
                valid_files = [
                    file for file in change.new.keys()
                    if file.lower().endswith(('.shp', '.geojson'))
                ]
                valid_file = valid_files[0]
                aoi_gdf = (gpd.read_file(valid_file).to_crs(
                    "EPSG:4326").explode().reset_index(drop=True))

                # Create a geodata
                geodata = GeoData(geo_dataframe=aoi_gdf,
                                  style={
                                      'color': 'black',
                                      'weight': 3
                                  })

                # Add to map
                xmin, ymin, xmax, ymax = aoi_gdf.total_bounds
                self.m.fit_bounds([[ymin, xmin], [ymax, xmax]])
                self.m.add_layer(geodata)

                # If completed, add to attribute
                self.gdf_uploaded = aoi_gdf

            except IndexError:
                print(
                    "Cannot read uploaded files. Please ensure that data is "
                    "in either GeoJSON or ESRI Shapefile format.",
                    end='\r')
                self.gdf_uploaded = None

            except fiona.errors.DriverError:
                print(
                    "Shapefile is invalid. Please ensure that all shapefile "
                    "components (e.g. .shp, .shx, .dbf, .prj) are uploaded.",
                    end='\r')
                self.gdf_uploaded = None

    def run_app(self, change):

        # Clear progress bar and output areas before running.
        self.status_info.clear_output()
        self.output_plot.clear_output()
        
        with self.status_info:
            # Load the area of interest from the map or uploaded files.
            if self.gdf_uploaded is not None:
                aoi_gdf = self.gdf_uploaded
            elif self.gdf_drawn is not None:
                aoi_gdf = self.gdf_drawn
            else:
                print(f'No valid polygon drawn on the map or uploaded. Please draw a valid a transect on the map, or upload a GeoJSON or ESRI Shapefile.',
                      end='\r')
                aoi_gdf = None

            # If valid area of interest data returned. Load the selected Global Forest Change data.
            if aoi_gdf is not None:
                
                if self.gfclayer_ds is None:
                    if self.gfclayer != "alllayers":
                        self.gfclayer_ds = load_gfclayer(gdf_drawn=aoi_gdf, gfclayer=self.gfclayer)
                    else:
                        self.gfclayer_ds = load_all_gfclayers(gdf_drawn=aoi_gdf)
                else:
                    print("Using previously loaded data")
                
                # Plot the selected Global Forest Change layer.
                if self.gfclayer_ds is not None:
                    with self.output_plot:
                        plot_gfclayer(gfclayer_ds=self.gfclayer_ds,
                                      start_year=self.start_year,
                                      end_year=self.end_year,
                                      gfclayer=self.gfclayer)
                else:
                    with self.status_info:
                        print(f"No Global Forest Change {self.gfclayer} layer data found in the selected area. Please select a new polygon over an area with data.")