import json
import bokeh
import rioxarray
import numpy as np
import panel as pn
import xarray as xr
import geoviews as gv
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import deafrica_tools.app.widgetconstructors as deawidgets

from io import BytesIO
from traitlets import Unicode
from IPython.display import display
from deafrica_tools.spatial import xr_rasterize
from deafrica_tools.spatial import reverse_geocode
from deafrica_tools.dask import create_local_dask_cluster
from ipywidgets import GridspecLayout, Button, Layout, HBox, VBox, HTML, Output
from ipyleaflet import (
    WMSLayer,
    basemaps,
    basemap_to_tiles,
    Map,
    DrawControl,
    WidgetControl,
    LayerGroup,
    LayersControl,
    GeoData,
)

# Load the bokeh extension.
gv.extension("bokeh", logo=False)

# Turn off all warnings.
import warnings

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


def get_basemap(basemap_url):
    """
    Gets the geoviews tile to use as a basemap based on a url
    """
    if basemap_url == "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png":
        basemap = gv.tile_sources.OSM
    elif (
        basemap_url
        == "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ):
        basemap = gv.tile_sources.EsriUSATopo

    return basemap


def plot_gfclayer(self):
    
    # Create the basemap.
    plot_basemap = get_basemap(self.basemap.url).opts(height=500, width=700)
    
    # Pass the ds xarray.Dataset to hv.Dataset
    # to create and object called "dataset."
    dataset = gv.Dataset(
        data=self.gfclayer_ds,
        kdims=list(self.gfclayer_ds.dims),
        vdims=self.gfclayer,
    )

    if self.gfclayer == "gain":
        # Color map to use to plot. 
        cmap = "Greens"
        # Ticks to be displayed on the colorbar.
        ticks = [1]
        ticker = bokeh.models.FixedTicker(ticks=ticks)
        # Ticklabels for the displayed ticks on the colorbar.
        ticklabels = ["gain"]
        major_label_overrides = dict(zip(ticks, ticklabels))
        
        # Pass the dataset to gv.image to create an object called "image" which is
        # an image element.
        # Elements are the simplest viewable components in HoloViews/GeoViews.
        image = dataset.to(gv.Image).opts(
            colorbar=True,
            cmap=cmap,
            title=f"Forest cover gain during the period 2000–2012",
            clabel="Global forest cover gain 2000–2012",
            colorbar_opts={
                "ticker": ticker,
                "major_label_overrides": major_label_overrides,
            },
            height=500,
            width=700,
        )
    
    if self.gfclayer == "treecover2000":
        # Color map to use to plot.
        cmap = "Greens"
        image = dataset.to(gv.Image).opts(
            colorbar=True,
            cmap=cmap,
            title=f"Tree cover in the year 2000",
            clabel="Percentage tree canopy cover for the year 2000",
            height=500,
            width=700,
        )
        
    if self.gfclayer == "lossyear":
        # Number of years from 2000 represented in the GFC lossyear.
        no_years = 21
        # Color map to use to plot.
        cmap = plt.get_cmap(name="gist_rainbow_r", lut=no_years)
        color_list = color_list = [mcolors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        # Location of transition from one color to the next.
        color_levels = list(np.arange(1 - 0.5, no_years + 1, 1))
        # Ticks to be displayed on the colorbar.
        ticks = list(range(1, 1 + no_years, 1))
        ticker = bokeh.models.FixedTicker(ticks=ticks)
        # Ticklabels for the displayed ticks on the colorbar.
        ticklabels = [str(i) for i in range(2001, 2001 + no_years, 1)]
        major_label_overrides = dict(zip(ticks, ticklabels))

        # Pass the dataset to gv.image to create an object called "image" which is
        # an image element.
        # Elements are the simplest viewable components in HoloViews/GeoViews.
        image = dataset.to(gv.Image).opts(
            colorbar=True,
            cmap=color_list,
            color_levels=color_levels,
            title=f"Forest loss during the period 2000–20{no_years}",
            clabel="Year of gross forest cover loss event",
            colorbar_opts={
                "ticker": ticker,
                "major_label_overrides": major_label_overrides,
            },
            height=500,
            width=700,
        )
  
    # Overlays are a collection of HoloViews objects to be displayed overlaid
    # on one another with the same axes.
    # Overlays are containers created by using the * operator on elements.
    overlay = plot_basemap * image
    # Convert the geoviews object to a displayable pane.
    map_pane = pn.panel(overlay)
    # Convert the pane to an ipywidget.
    map_widget = pn.ipywidget(map_pane)
    map_widget.layout = make_box_layout()
    return map_widget


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
            ("Tree canopy cover for the year 2000", "treecover2000")
        ]
        # Set the default GFC layer to be plotted / initial value for the widget.
        self.gfclayer = self.gfclayers_list[0][1]
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
            [basemap_selection, gfclayer_selection, enable_max_size]
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
        grid_columns = 10
        grid_height = "1500px"
        grid_width = "auto"
        grid = GridspecLayout(
            grid_rows, grid_columns, height=grid_height, width=grid_width
        )

        # Place app widgets and components in app layout.
        # [rows, columns]
        grid[0, :] = self.header
        grid[1:4, 0:3] = parameter_selection
        grid[4, 0:3] = run_button
        grid[5:, 0:3] = self.status_info
        grid[4:, 3:] = self.output_plot
        grid[1:4, 3:] = self.m
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

    def update_checkbox_max_size(self, change):
        """
        Sets the value of self.max_size to True when the
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
                    map_widget = plot_gfclayer(self)
                    print(f"Plotting Global Forest Change {self.gfclayer} layer:")
                    display(map_widget)
            else:
                with self.status_info:
                    print(
                        f"""No Global Forest Change {self.gfclayer} layer 
                              data found in the selected area. Please select 
                              a new polygon over an area with data."""
                    )

        # If no valid polygon was drawn.
        else:
            with self.status_info:
                print(
                    'Please draw a valid polygon on the map, then click on "Geneate plot"'
                )