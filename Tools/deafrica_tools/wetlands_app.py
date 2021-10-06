# wetlands_app.py

"""
Description: This file contains a set of python functions for the interactive wetlands insight tool

License: The code in this notebook is licensed under the Apache License,
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

Functions included:
    

Last modified: Oct 2021

"""

#Import required packages
import datacube
import seaborn as sns
import matplotlib.pyplot as plt
from datacube.utils.geometry import CRS
from ipyleaflet import WMSLayer, basemaps, basemap_to_tiles, Map, DrawControl, WidgetControl, LayerGroup
from traitlets import Unicode
from ipywidgets import GridspecLayout, Button, Layout, jslink, IntText, IntSlider, DatePicker, HBox, VBox, Text, BoundedFloatText, HTML, Dropdown, Output, Label
import json
import geopandas as gpd
from io import BytesIO
from dask.diagnostics import ProgressBar

from deafrica_tools.dask import create_local_dask_cluster
from deafrica_tools.wetlands import WIT_drill

def create_map():
    
    basemap_osm = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    basemap_esri = basemap_to_tiles(basemaps.Esri.WorldImagery)
    basemap_cartodb = basemap_to_tiles(basemaps.CartoDB.Positron)
    
    m = Map(center=(4, 20), zoom=3, basemap=basemap_esri)
    
    return m

def create_deafrica_layer(product, date):
    
    # Load DEA WMS
    class TimeWMSLayer(WMSLayer):
        time = Unicode("").tag(sync=True, o=True)

    time_wms = TimeWMSLayer(
        url="https://ows.digitalearth.africa/",
        layers=product,
        time=date,
        format="image/png",
        transparent=True,
        attribution="Digital Earth Africa",
    )
    
    return time_wms

def create_datepicker(description):
    
    date_picker = DatePicker(
        description=description,
        disabled=False
    )
    
    return date_picker

def create_drawcontrol():
    
    draw_control = DrawControl()
    

    draw_control.rectangle = {
        "shapeOptions": {
            "fillColor": "#fca45d",
            "color": "#fca45d",
            "fillOpacity": 1.0
        }
    }
    
    draw_control.polygon = {
        "shapeOptions": {
            "fillColor": "#6be5c3",
            "color": "#6be5c3",
            "fillOpacity": 1.0
        },
        "drawError": {
            "color": "#dd253b",
            "message": "Error!"
        },
        "allowIntersection": False
    }
    
    # Disable other forms
    draw_control.marker={}
    draw_control.circle={}
    draw_control.circlemarker={}
    draw_control.polyline={}
    
    
    return draw_control

def create_inputtext(value, placeholder, description):
    
    input_text = Text(
        value=value,
        placeholder=placeholder,
        description=description,
        disabled=False
    )
    
    return input_text
    
def create_boundedfloattext(value, description, min_val, max_val, step_val):
    
    float_text = BoundedFloatText(
        value=value,
        min=min_val,
        max=max_val,
        step=step_val,
        description=description,
        disabled=False
    )
    
    return float_text

def create_html(value):
    
    html = HTML(
        value=value,
    )
    
    return html

def create_dropdown(options, value, description):
    
    dropdown = Dropdown(
        options=options,
        value=value,
        description=description,
    )
    
    return dropdown

def make_box_layout():
     return Layout(
        margin='0px 15px 15px 0px',
        padding='5px 5px 5px 5px'
     )

def create_expanded_button(description, button_style):
    return Button(description=description, button_style=button_style, layout=Layout(width='auto', height='auto'))

class wit_app(HBox):
    
    def __init__(self):
        super().__init__()
        
        ##########################################################
        
        # set any initial attributes here 
        self.startdate = '2020-01-01'
        self.enddate = '2020-03-01'
        self.mingooddata = 0.0
        self.resamplingfreq = '1M'
        self.out_csv = 'example_WIT.csv'
        self.out_plot = 'example_WIT.png'
        self.product_list = [('None', 'none'), ('Sentinel-2 Geomedian', 'gm_s2_annual'), ('Water Observations from Space', 'wofs_ls_summary_annual')]
        self.product = self.product_list[0][1]
        self.product_year = '2020-01-01'
        self.target = None
        self.action = None
        self.gdf_drawn = None
        
        self.paramlog = create_html('<h3>Wetlands Insight Tool</h3><p>Select parameters and AOI</p>')
        self.paramlog.layout = make_box_layout()
        
        def update_geojson(target, action, geo_json):
        
            self.action = action
            
            json_data = json.dumps(geo_json)
            binary_data = json_data.encode()
            io = BytesIO(binary_data)
            io.seek(0)
            
            gdf = gpd.read_file(io)
            gdf.crs = "EPSG:4326"
            self.gdf_drawn = gdf
            
            gdf_drawn_epsg6933 = gdf.copy().to_crs("EPSG:6933")
            m2_per_km2 = 10**6
            area = gdf_drawn_epsg6933.area.values[0]/m2_per_km2
            base_text = f'<h3>Wetlands Insight Tool</h3><p><b>Total polygon area</b>: {area:.2f} km<sup>2</sup></p>'
            
            if area <= 10000:
                confirmation_text = f'<p style="color:#33cc33;">Area falls within recommended limit</p>'
                self.paramlog.value = base_text + confirmation_text
            else:
                warning_text = f'<p style="color:#ff5050;">Area is too large, please update your polygon</p>'
                self.paramlog.value = base_text + warning_text

        
        ##########################################################
        
        self.progress_bar = Output(layout=make_box_layout())
        self.wit_plot = Output(layout=make_box_layout())
        self.progress_header = create_html('')
        
        ##########################################################
        
        self.deafrica_layers = LayerGroup(layers=())
        
        # Create map widget
        self.m = create_map()
        draw_control = create_drawcontrol()
        self.m.add_control(draw_control)
        self.basemap = self.m.basemap
        self.m.add_layer(self.deafrica_layers)
        self.m.layout = make_box_layout()
        
        ##########################################################
        
        # Create parameter widgets
        startdate_picker = create_datepicker('')
        enddate_picker = create_datepicker('')
        min_good_data = create_boundedfloattext(self.mingooddata, '', 0.0, 1.0, 0.05)
        resampling_freq = create_inputtext(self.resamplingfreq, self.resamplingfreq, '')
        output_csv = create_inputtext(self.out_csv, self.out_csv, '')
        output_plot = create_inputtext(self.out_plot, self.out_plot, '')
        basemap_dropdown = create_dropdown(self.product_list, self.product_list[0][1], '')
        run_button = create_expanded_button('Run', 'info')
        
        parameter_selection = VBox(
            [
                HTML("<b>Start Date:</b>"),
                startdate_picker, 
                HTML("<b>End Date:</b>"),
                enddate_picker,
                HTML("<b>Minimum Good Data:</b>"),
                min_good_data,
                HTML('<b>Resampling Frequency:</b>'),
                resampling_freq,
                HTML("<b>Output CSV:</b>"),
                output_csv,
                HTML("<b>Output Plot:</b>"),
                output_plot,
                HTML("<b>DEA Overlay:</b>"),
                basemap_dropdown,
                run_button,
            ]
        )
        parameter_selection.layout = make_box_layout()
        
        ##########################################################
        
        # Create the layout #[rowspan, colspan]
        grid = GridspecLayout(12, 12, height='1200px', width='auto')
        
        # Controls and Status
        grid[0, 0:6] = self.paramlog
        grid[1:7, 0:3] = parameter_selection
        
        # Map
        grid[1:7, 3:] = self.m
        
        # Progress bar
        grid[7, 0:3] = self.progress_header
        grid[8:, 0:3] = self.progress_bar

        # Plot
        grid[7:, 3:] = self.wit_plot
        
        # Display using HBox children attribute
        self.children = [grid]
        
        ##########################################################
        
        # Run update functions whenever various widgets are changed.
        startdate_picker.observe(self.update_startdate, 'value')
        enddate_picker.observe(self.update_enddate, 'value')
        min_good_data.observe(self.update_mingooddata, 'value')
        resampling_freq.observe(self.update_resamplingfreq, 'value')
        output_csv.observe(self.update_outputcsv, 'value')
        output_plot.observe(self.update_outputplot, 'value')
        basemap_dropdown.observe(self.update_basemap, 'value')
        run_button.on_click(self.run_app)
        draw_control.on_draw(update_geojson)
        
        ##########################################################
        
    # set the start date to the new edited date
    def update_startdate(self, change):
        self.startdate = change.new
        
    # set the end date to the new edited date    
    def update_enddate(self, change):
        self.enddate = change.new 
    
    # set the min good data
    def update_mingooddata(self, change):
        self.mingooddata = change.new 
        
    # set the resampling frequency
    def update_resamplingfreq(self, change):
        self.resamplingfreq = change.new 
        
    # set the output csv
    def update_outputcsv(self, change):
        self.out_csv = change.new 
        
    # set the output plot
    def update_outputplot(self, change):
        self.out_plot = change.new
        
    # Update product
    def update_basemap(self, change):
        
        self.product = change.new
        
        if self.product == 'none':
            self.deafrica_layers.clear_layers()
        else:
            self.deafrica_layers.clear_layers()
            layer = create_deafrica_layer(self.product, self.product_year)
            self.deafrica_layers.add_layer(layer)
        
    def run_app(self, change):
        
        # Clear progress bar and output areas before running
        self.progress_bar.clear_output()
        self.wit_plot.clear_output()
        
        # Connect to datacube database
        dc = datacube.Datacube(app="wetland_app")

        # Configure local dask cluster
        #create_local_dask_cluster()

        # Set any defaults
        resample_frequency = '1M'
        TCW_threshold = -0.035
        dask_chunks = dict(x=1000, y=1000, time=1)
        
        
        self.progress_header.value = f'<h3>Progress</h3>'
        
        with self.progress_bar:
            print("running WIT")
        
        #self.paramlog.value = 'Running WIT'
        # run wetlands polygon drill
        
        with self.progress_bar:
            with ProgressBar():
                df = WIT_drill(
                    gdf=self.gdf_drawn,
                    time=(self.startdate, self.enddate),
                    min_gooddata=self.mingooddata,
                    resample_frequency=resample_frequency,
                    TCW_threshold=TCW_threshold,
                    export_csv=self.out_csv,
                    dask_chunks=dask_chunks,
                    verbose=False,
                    verbose_progress=True,
                )
        
        #self.progress_bar.clear_output()
        with self.progress_bar:
            print("WIT complete")
        #self.paramlog.value = 'WIT Complete'

        # save the csv
        if self.out_csv:
            df.to_csv(self.out_csv, index_label="Datetime")

        # ---Plotting------------------------------

        with self.wit_plot:

            fontsize = 17
            plt.rcParams.update({'font.size': fontsize})
            # set up color palette
            pal = [
                sns.xkcd_rgb["cobalt blue"],
                sns.xkcd_rgb["neon blue"],
                sns.xkcd_rgb["grass"],
                sns.xkcd_rgb["beige"],
                sns.xkcd_rgb["brown"],
            ]

            # make a stacked area plot
            plt.close("all")
            
            fig, ax = plt.subplots(constrained_layout=True, figsize=(22, 6))
            
            ax.stackplot(
                df.index,
                df.wofs_area_percent,
                df.wet_percent,
                df.green_veg_percent,
                df.dry_veg_percent,
                df.bare_soil_percent,
                labels=[
                    "open water",
                    "wet",
                    "green veg",
                    "dry veg",
                    "bare soil",
                ],
                colors=pal,
                alpha=0.6,
            )
            

            # set axis limits to the min and max
            ax.set_ylim(0, 100)
            ax.set_xlim(df.index[0], df.index[-1])
            ax.tick_params(axis="x", labelsize=fontsize)
            
            # add a legend and a tight plot box
            ax.legend(loc="lower left", framealpha=0.6)
            ax.set_title("Fractional Cover, Wetness, and Water")
            #plt.tight_layout()
            plt.show()
            
            if self.out_plot:
                # save the figure
                fig.savefig(f"{self.out_plot}")