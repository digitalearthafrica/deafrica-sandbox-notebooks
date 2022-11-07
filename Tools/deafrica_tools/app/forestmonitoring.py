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
    
    
def load_gfclayer(self):
    """
    Loads the selected Global Forest Change layer for the
    area drawn on the map widget.
    """
    # Configure local dask cluster.
    client = create_local_dask_cluster(return_client=True, display_client=True)

    # Get the coordinates of the top-left corner for each Global Forest Change tile,
    # covering the area of interest.
    min_lat, max_lat = (
        self.gdf_drawn.bounds.miny.item(),
        self.gdf_drawn.bounds.maxy.item(),
    )
    min_lon, max_lon = (
        self.gdf_drawn.bounds.minx.item(),
        self.gdf_drawn.bounds.maxx.item(),
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
    base_url = f"https://storage.googleapis.com/earthenginepartners-hansen/GFC-2021-v1.9/Hansen_GFC-2021-v1.9_{self.gfclayer}_"
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
    if self.gfclayer == "lossyear":
        ds = ds.where(ds != 0)
    # Mask pixels representing no gain (encoded as 0) in the "gain" layer.
    elif self.gfclayer == "gain":
        ds = ds.where(ds != 0)
    # Mask pixels with 0 percentage tree canopy cover.
    elif self.gfclayer == "treecover2000":
        ds = ds.where(ds != 0)

    # Create a mask from the area of interest GeoDataFrame.
    mask = xr_rasterize(self.gdf_drawn, ds)
    # Mask the dataset.
    ds = ds.where(mask)
    # Convert the xarray.DataArray to a dataset.
    ds = ds.to_dataset(name=self.gfclayer)
    # Compute.
    ds = ds.compute()
    # Assign the "EPSG:4326" CRS to the dataset.
    ds.rio.write_crs(4326, inplace=True)
    ds = ds.transpose("latitude", "longitude")

    # Close down the dask client.
    client.close()
    return ds


def plot_gfclayer(self):

    if self.gfclayer == "gain":
        ds = self.gfclayer_ds[self.gfclayer]

        # Define some plotting parameters.
        color = "#6CAE75"
        figure_width = 10
        figure_length = 10
        title = f"Forest Cover Gain from 2000 to 2012"

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
            data={"Forest Cover Gain in km$^2$": [counts_dict[1.0] * per_pixel_area]}
        )
        # Get the total area.
        print_statement = f'Total Forest Cover Gain from 2000 to 2012: {round(df["Forest Cover Gain in km$^2$"].sum(), 2)} km2'
        
        # Plot the figure.
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
        plt.show()
        
        print(print_statement)
        
        ## Export the results.
        file_name = f"forest_cover_gain_from_2000_to_2012"
        print(f"\nExporting results as: \n\t{file_name}.csv and \n\t{file_name}.png")
        # Export the results as a csv file.
        df.to_csv(f"{file_name}.csv", index=False)
        # Save the plot.
        plt.savefig(f"{file_name}.png")
        
    if self.gfclayer == "treecover2000":
        ds = self.gfclayer_ds[self.gfclayer]

        # Define some plotting parameters.
        figure_width = 10
        figure_length = 10
        title = f"Tree Canopy Cover for the Year 2000"

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
            data={"Tree Cover in km$^2$": [counts_dict[1.0] * per_pixel_area]}
        )
        # Get the total area.
        print_statement = f'Total Forest Cover in 2000: {round(df["Tree Cover in km$^2$"].sum(), 2)} km2'
    
        # Plot the figure.
        fig, ax = plt.subplots(figsize=(figure_width, figure_length))
        im = ds.plot(cmap="Greens", add_colorbar=False, ax=ax)
        # Add a colorbar to the plot.
        cbar = plt.colorbar(mappable=im)
        cbar.set_label(
            "Percentage tree canopy cover for year 2000", labelpad=-65, y=0.25
        )
        # Add a title to the plot.
        plt.title(title)
        plt.show()

        print(print_statement)
        
        ## Export the results.
        file_name = f"forest_cover_in_2000"
        print(f"\nExporting results as: \n\t{file_name}.csv and \n\t{file_name}.png")
        # Export the results as a csv file.
        df.to_csv(f"{file_name}.csv", index=False)
        # Save the plot.
        plt.savefig(f"{file_name}.png")
        
    if self.gfclayer == "lossyear":
        ds = self.gfclayer_ds[self.gfclayer]

        # Mask the dataset to the selected time range.
        selected_years = list(range(self.start_year, self.end_year + 1))
        mask = ds.isin(selected_years)
        ds = ds.where(mask)

        ## Get the area of loss for the selected time range.
        selected_years_str = [str(2000 + i) for i in selected_years]
        selected_years_dict = dict(zip(selected_years, selected_years_str))
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
        area_dict = {}
        for k, v in counts_dict.items():
            area_dict[selected_years_dict[k]] = v * per_pixel_area

        # Save the results as a pandas DataFrame.
        df = pd.DataFrame(
            data={
                "Year": area_dict.keys(),
                "Forest Cover Loss in km$^2$": area_dict.values(),
            }
        )
        # Get the total area.
        print_statement = rf'Total Forest Cover Loss from {df["Year"].min()} to {df["Year"].max()}: {round(df["Forest Cover Loss in km$^2$"].sum(), 2)} km2'

        ## Define some plotting parameters.
        figure_width = 10
        figure_length = 15
        nrows = 2
        ncols = 1
        title = f"Forest Cover Loss from {selected_years_str[0]} to {selected_years_str[-1]}"
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
        cmap = mcolors.ListedColormap(colors=color_list, N=len(selected_years))
        # Location of transition from one color to the next.
        color_levels = list(np.arange(self.start_year - 0.5, self.end_year + 1, 1))
        norm = mcolors.BoundaryNorm(boundaries=color_levels, ncolors=cmap.N)

        # Plot the figure.
        fig, (ax1, ax2) = plt.subplots(
            nrows, ncols, figsize=(figure_width, figure_length)
        )
        im = ds.plot(ax=ax1, cmap=cmap, norm=norm, add_colorbar=False)
        # Add a title to the subplot.
        ax1.set_title(title)
        # Add a colorbar to the subplot.
        cbar = plt.colorbar(mappable=im, ticks=selected_years)
        cbar.set_label("Year of gross forest cover loss event", labelpad=-60, y=0.25)
        cbar.set_ticklabels(selected_years_str)
        # Plot the second subplot.
        df.plot(
            x="Year",
            y="Forest Cover Loss in km$^2$",
            ylabel="Forest Cover Loss in km$^2$",
            title=title,
            ax=ax2,
        )
        plt.show()
        
        print(print_statement)
        
        ## Export the results.
        file_name = f"forest_cover_loss_from_{selected_years_str[0]}_to_{selected_years_str[-1]}"
        print(f"\nExporting results as: \n\t{file_name}.csv and \n\t{file_name}.png")
        # Export the results as a csv file.
        df.to_csv(f"{file_name}.csv", index=False)
        # Save the plot.
        plt.savefig(f"{file_name}.png")
        
        
class forest_monitoring_app(HBox):
    def __init__(self):
        super().__init__()

        ##################
        # HEADER FOR APP #
        ##################

        # Create the header widget.
        header_title_text = "<h3>Digital Earth Africa Forest Change</h3>"
        instruction_text = """<p>Select the desired Global Forest Change layer, then zoom in and draw a polygon to
                                select an area for which to plot the selected Global Forest Change layer.</p>"""
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

        ## Put the app controls widgets into a single container.
        parameter_selection = VBox(
            [
                basemap_selection,
                gfclayer_selection,
                timerange_selection,
                enable_max_size,
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
        grid[1:5, 0:4] = parameter_selection
        grid[5, 0:4] = run_button
        grid[6:, 0:4] = self.status_info
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
        Sets the value of self.max_size to True when tcolor_selectionhe
        checkbox_max_size CheckBox is checked.
        """
        self.max_size = change.new

    def run_app(self, change):

        # Clear progress bar and output areas before running.
        self.status_info.clear_output()
        self.output_plot.clear_output()

        # Verify that the polygon was drawn.
        if self.gdf_drawn is not None:
            # Load the seleced Global Forest Change layer
            # and add it to the self.gfclayer_ds attribute.
            with self.status_info:
                if self.gfclayer_ds is None:
                    self.gfclayer_ds = load_gfclayer(self)
                else:
                    print("Using previously loaded data")

            # Plot the selected Global Forest Change layer.
            if self.gfclayer_ds is not None:
                with self.output_plot:
                    ## Check if the dataset is empty.
                    array = self.gfclayer_ds[self.gfclayer].values
                    # Check if the array is entirely composed of nans.
                    condition = np.all(np.isnan(array))
                    # If the condition is not true (dataset is not empty) plot the layer.
                    if not condition:
                        plot_gfclayer(self)
                    else:
                        print(
                            f"No Global Forest Change {self.gfclayer} layer data found in the selected area. Please select a new polygon over an area with data."
                        )
            else:
                with self.status_info:
                    print(
                        f"No Global Forest Change {self.gfclayer} layer data found in the selected area. Please select a new polygon over an area with data."
                        ""
                    )

        # If no valid polygon was drawn.
        else:
            with self.status_info:
                print(
                    'Please draw a valid polygon on the map, then click on "Geneate plot"'
                )