# areaofinterest.py
"""
Upload area of interest as a GeoJSON or ESRI Shapefile and draw it on a map 
"""

import geopandas as gpd
import ipywidgets as widgets
import deafrica_tools.app.widgetconstructors as deawidgets
from ipyleaflet import GeoData
from ipywidgets import HBox, Layout, VBox, Output, GridspecLayout


def make_box_layout():
    """
    Defines a number of CSS properties that impact how a widget is laid out.
    """
    return Layout(
        margin="0px 10px 10px 0px",
        padding="5px 5px 5px 5px",
        width="100%",
        height="100%",
    )


class upload_aoi(HBox):
    def __init__(self):
        super().__init__()
        ##################
        # HEADER FOR APP #
        ##################

        # Add widget to enable uploading a geojson or ESRI shapefile.
        self.gdf_uploaded = None
        fileupload_aoi = widgets.FileUpload(accept="", multiple=True)
        # Register the update function to be called for the file upload.
        fileupload_aoi.observe(self.update_fileupload_aoi, "value")
        fileupload_html = deawidgets.create_html(
            value=f"""<h2>Upload a GeoJSON or ESRI Shapefile</h2></br><i></br>Upload a GeoJSON or ESRI Shapefile (<5 mb) containing a single area of interest. When uploading an ESRI shapefile, ensure that all shapefile components (e.g. .shp, .shx, .dbf, .prj) are uploaded.</i>"""
        )
        fileupload = VBox([fileupload_aoi])

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
            map_center=(6.587292, 1.532833),
            zoom_level=3,
        )
        self.m.layout = make_box_layout()

        ###############################
        # SPECIFICATION OF APP LAYOUT #
        ###############################

        # Create the app layout.
        grid_rows = 6
        grid_columns = 10
        grid_height = "600px"
        grid_width = "auto"
        grid = GridspecLayout(
            grid_rows, grid_columns, height=grid_height, width=grid_width
        )

        # Place app widgets and components in app layout.
        # [rows, columns]
        grid[1:2, 0:3] = fileupload_html
        grid[2:3, 0:3] = fileupload
        grid[3:6, 0:3] = self.status_info
        grid[:6, 3:] = self.m
        # Display using HBox children attribute
        self.children = [grid]

    def update_fileupload_aoi(self, change):
        # Save to file
        for uploaded_filename in change.new.keys():
            with open(uploaded_filename, "wb") as output_file:
                content = change.new[uploaded_filename]["content"]
                output_file.write(content)

        with self.status_info:

            try:

                valid_files = [
                    file
                    for file in change.new.keys()
                    if file.lower().endswith((".shp", ".geojson"))
                ]
                valid_file = valid_files[0]
                aoi_gdf = (
                    gpd.read_file(valid_file)
                    .to_crs("EPSG:4326")
                    .explode()
                    .reset_index(drop=True)
                )

                # Create a geodata
                geodata = GeoData(
                    geo_dataframe=aoi_gdf, style={"color": "black", "weight": 3}
                )
                # Add to map
                xmin, ymin, xmax, ymax = aoi_gdf.total_bounds
                self.m.fit_bounds([[ymin, xmin], [ymax, xmax]])
                self.m.add_layer(geodata)

                # Latitude and longitude range class variables
                upload_aoi.lon_range = (xmin, xmax)
                upload_aoi.lat_range = (ymin, ymax)

                # If completed, add to attribute
                self.gdf_uploaded = aoi_gdf
                self.status_info.clear_output()
                self.output_plot.clear_output()

                with self.status_info:
                    # Load the area of interest from uploaded files.
                    if self.gdf_uploaded is not None:
                        aoi_gdf = self.gdf_uploaded

                    # If valid area of interest data returned.
                    if aoi_gdf is not None:
                        print("Area of interest successfully loaded")

            except IndexError:
                print(
                    "Cannot read uploaded files. Please ensure that data is "
                    "in either GeoJSON or ESRI Shapefile format.",
                    end="\r",
                )
                self.gdf_uploaded = None

            except fiona.errors.DriverError:
                print(
                    "Shapefile is invalid. Please ensure that all shapefile "
                    "components (e.g. .shp, .shx, .dbf, .prj) are uploaded.",
                    end="\r",
                )
