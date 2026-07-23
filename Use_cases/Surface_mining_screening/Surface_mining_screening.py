import os
import zipfile
import tempfile
from pathlib import Path
from datetime import date
import calendar

import geopandas as gpd
import ipywidgets as widgets
from IPython.display import display, clear_output
from shapely.geometry import Point


# ---------------------------------------------------------------------
# AOI helper functions
# ---------------------------------------------------------------------

def create_aoi_from_latlon(lat, lon, buffer_km=5, output_crs="EPSG:6933"):
    """
    Create an AOI polygon from latitude, longitude, and buffer distance.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    buffer_km : float, optional
        Buffer distance in kilometres. Default is 5 km.
    output_crs : str, optional
        Projected CRS used for buffering. Default is EPSG:6933.

    Returns
    -------
    geopandas.GeoDataFrame
        Buffered AOI polygon in EPSG:4326.
    """

    if lat < -90 or lat > 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if lon < -180 or lon > 180:
        raise ValueError("Longitude must be between -180 and 180.")

    if buffer_km <= 0:
        raise ValueError("Buffer distance must be greater than 0 km.")

    point_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)],
        crs="EPSG:4326"
    )

    projected = point_gdf.to_crs(output_crs)
    buffered = projected.buffer(buffer_km * 1000)

    aoi = gpd.GeoDataFrame(
        geometry=buffered,
        crs=output_crs
    ).to_crs("EPSG:4326")

    return aoi


# ---------------------------------------------------------------------
# File upload helper
# ---------------------------------------------------------------------

def _get_uploaded_file(upload_widget):
    """
    Safely extract uploaded file information from ipywidgets FileUpload.

    This handles both older and newer ipywidgets formats.
    """

    if not upload_widget.value:
        raise ValueError("No shapefile uploaded.")

    value = upload_widget.value

    # ipywidgets 7 format: dict
    if isinstance(value, dict):
        uploaded_file = list(value.values())[0]
        file_name = uploaded_file.get("metadata", {}).get("name")
        content = uploaded_file.get("content")

    # ipywidgets 8 format: tuple/list of dict-like objects
    elif isinstance(value, (tuple, list)):
        uploaded_file = value[0]
        file_name = uploaded_file.get("name") or uploaded_file.get("metadata", {}).get("name")
        content = uploaded_file.get("content")

    else:
        raise ValueError("Unsupported upload format from FileUpload widget.")

    if file_name is None or content is None:
        raise ValueError("Could not read uploaded file name or content.")

    return file_name, content


def read_uploaded_shapefile(upload_widget):
    """
    Read an uploaded zipped shapefile from an ipywidgets FileUpload widget.

    The shapefile must be uploaded as a .zip containing at least:
    .shp, .shx, .dbf, and .prj

    Parameters
    ----------
    upload_widget : ipywidgets.FileUpload
        File upload widget.

    Returns
    -------
    geopandas.GeoDataFrame
        AOI from uploaded shapefile in EPSG:4326.
    """

    file_name, content = _get_uploaded_file(upload_widget)

    if not file_name.lower().endswith(".zip"):
        raise ValueError("Please upload the shapefile as a zipped .zip file.")

    temp_dir = tempfile.mkdtemp(prefix="uploaded_aoi_")
    zip_path = os.path.join(temp_dir, file_name)

    with open(zip_path, "wb") as f:
        f.write(content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    shp_files = list(Path(temp_dir).rglob("*.shp"))

    if len(shp_files) == 0:
        raise ValueError("No .shp file found inside the uploaded zip.")

    gdf = gpd.read_file(shp_files[0])

    if gdf.empty:
        raise ValueError("The uploaded shapefile is empty.")

    if gdf.crs is None:
        raise ValueError("The shapefile has no CRS. Please define its projection first.")

    return gdf.to_crs("EPSG:4326")


# ---------------------------------------------------------------------
# Product and statistic helper functions
# ---------------------------------------------------------------------

def get_product_mapping():
    """
    Product name mapping.

    Update these names to match the exact product names in your ODC environment.

    Returns
    -------
    dict
        User-facing product labels mapped to ODC product names.
    """

    return {
        "Annual GeoMAD": "gm_ls_annual",
        "Semi-annual GeoMAD": "gm_ls_semiannual",
        "Sentinel imagery": "s2_l2a_c1"
    }


def get_sentinel_stat_options():
    """
    Return statistics available for Sentinel imagery.
    """

    return [
        "Median",
        "Mean",
        "Minimum",
        "Maximum",
        "Standard deviation",
        "Geomedian"
    ]


def build_query(
    aoi,
    selected_products,
    start_year,
    start_month,
    end_year,
    end_month,
    sentinel_stat=None,
    output_crs="EPSG:6933",
    resolution=(-30, 30),
    group_by="solar_day"
):
    """
    Build a query dictionary for Open Data Cube loading.

    Parameters
    ----------
    aoi : geopandas.GeoDataFrame
        AOI polygon in EPSG:4326.
    selected_products : list
        User-facing product labels selected from the UI.
    start_year, start_month, end_year, end_month : int
        Date range.
    sentinel_stat : str, optional
        Statistic selected for Sentinel imagery.
    output_crs : str, optional
        Output CRS for ODC loading.
    resolution : tuple, optional
        Output spatial resolution.
    group_by : str, optional
        ODC group_by option.

    Returns
    -------
    dict
        Query dictionary.
    """

    if not selected_products:
        raise ValueError("Please select at least one product.")

    start_date = date(int(start_year), int(start_month), 1)
    end_date = date(int(end_year), int(end_month), 28)

    if start_date > end_date:
        raise ValueError("Start date must be before end date.")

    product_mapping = get_product_mapping()

    unknown_products = [p for p in selected_products if p not in product_mapping]
    if unknown_products:
        raise ValueError(f"Unknown products selected: {unknown_products}")

    minx, miny, maxx, maxy = aoi.total_bounds

    query = {
        "products": [product_mapping[p] for p in selected_products],
        "selected_product_labels": selected_products,
        "time": (start_date.isoformat(), end_date.isoformat()),
        "x": (float(minx), float(maxx)),
        "y": (float(miny), float(maxy)),
        "crs": "EPSG:4326",
        "output_crs": output_crs,
        "resolution": resolution,
        "group_by": group_by
    }

    if "Sentinel imagery" in selected_products:
        if sentinel_stat is None:
            raise ValueError("Please select a Sentinel imagery statistic.")
        query["sentinel_statistic"] = sentinel_stat

    return query


def apply_sentinel_statistic(ds, statistic):
    """
    Apply a selected statistic to a Sentinel xarray Dataset/DataArray.

    This is optional and can be called after loading Sentinel imagery.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Sentinel imagery with a time dimension.
    statistic : str
        One of the Sentinel statistic options.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Dataset/DataArray reduced across time.
    """

    if "time" not in ds.dims:
        raise ValueError("The Sentinel dataset must have a 'time' dimension.")

    if statistic == "Median":
        return ds.median(dim="time", skipna=True)

    if statistic == "Mean":
        return ds.mean(dim="time", skipna=True)

    if statistic == "Minimum":
        return ds.min(dim="time", skipna=True)

    if statistic == "Maximum":
        return ds.max(dim="time", skipna=True)

    if statistic == "Standard deviation":
        return ds.std(dim="time", skipna=True)

    if statistic == "Geomedian":
        try:
            from odc.algo import xr_geomedian
        except ImportError as exc:
            raise ImportError(
                "Geomedian requires odc.algo. Please install/import odc.algo first."
            ) from exc

        return xr_geomedian(ds)

    raise ValueError(f"Unsupported Sentinel statistic: {statistic}")


# ---------------------------------------------------------------------
# Main UI function
# ---------------------------------------------------------------------

def display_loader_ui():
    """
    Display an interactive Jupyter Notebook UI for selecting AOI, products,
    date range, and Sentinel imagery statistics.

    Returns
    -------
    None
        Displays the UI directly in the notebook.
    """

    # -----------------------------
    # AOI widgets
    # -----------------------------
    aoi_mode = widgets.ToggleButtons(
        options=["Upload shapefile", "Lat/Lon buffer"],
        description="AOI:",
        button_style=""
    )

    shapefile_upload = widgets.FileUpload(
        accept=".zip",
        multiple=False,
        description="Upload .zip"
    )

    lat_input = widgets.FloatText(
        value=5.6,
        description="Latitude:",
        layout=widgets.Layout(width="250px")
    )

    lon_input = widgets.FloatText(
        value=-0.2,
        description="Longitude:",
        layout=widgets.Layout(width="250px")
    )

    buffer_input = widgets.FloatText(
        value=5,
        description="Buffer km:",
        layout=widgets.Layout(width="250px")
    )

    # -----------------------------
    # Product widgets
    # -----------------------------
    product_select = widgets.SelectMultiple(
        options=list(get_product_mapping().keys()),
        value=("Annual GeoMAD",),
        description="Products:",
        layout=widgets.Layout(width="320px", height="115px")
    )

    # -----------------------------
    # Sentinel statistics widget
    # -----------------------------
    sentinel_stat_select = widgets.Dropdown(
        options=get_sentinel_stat_options(),
        value="Median",
        description="Statistic:",
        layout=widgets.Layout(width="320px")
    )

    sentinel_stat_box = widgets.VBox([
        widgets.HTML("<b>Sentinel imagery statistic</b>"),
        sentinel_stat_select,
        widgets.HTML(
            "<small>This option appears only when Sentinel imagery is selected.</small>"
        )
    ])
    sentinel_stat_box.layout.display = "none"

    # -----------------------------
    # Date widgets
    # -----------------------------
    current_year = date.today().year
    years = list(range(2000, current_year + 2))
    months = list(range(1, 13))

    start_year = widgets.Dropdown(
        options=years,
        value=max(2000, current_year - 5),
        description="Start year:",
        layout=widgets.Layout(width="250px")
    )

    start_month = widgets.Dropdown(
        options=months,
        value=1,
        description="Start month:",
        layout=widgets.Layout(width="250px")
    )

    end_year = widgets.Dropdown(
        options=years,
        value=current_year,
        description="End year:",
        layout=widgets.Layout(width="250px")
    )

    end_month = widgets.Dropdown(
        options=months,
        value=12,
        description="End month:",
        layout=widgets.Layout(width="250px")
    )

    # -----------------------------
    # Buttons and output
    # -----------------------------
    run_button = widgets.Button(
        description="Run Load Query",
        button_style="success",
        icon="play"
    )

    clear_button = widgets.Button(
        description="Clear Output",
        button_style="warning",
        icon="trash"
    )

    output = widgets.Output()

    # -----------------------------
    # AOI boxes
    # -----------------------------
    shapefile_box = widgets.VBox([
        widgets.HTML("<b>Upload zipped shapefile</b>"),
        shapefile_upload,
        widgets.HTML(
            "<small>Zip must contain .shp, .shx, .dbf, and .prj files.</small>"
        )
    ])

    latlon_box = widgets.VBox([
        widgets.HTML("<b>Enter latitude, longitude, and buffer</b>"),
        lat_input,
        lon_input,
        buffer_input
    ])

    dynamic_aoi_box = widgets.VBox([shapefile_box])

    # -----------------------------
    # UI update functions
    # -----------------------------
    def update_aoi_box(change=None):
        if aoi_mode.value == "Upload shapefile":
            dynamic_aoi_box.children = [shapefile_box]
        else:
            dynamic_aoi_box.children = [latlon_box]

    def update_sentinel_stat_box(change=None):
        selected_products = list(product_select.value)

        if "Sentinel imagery" in selected_products:
            sentinel_stat_box.layout.display = "block"
        else:
            sentinel_stat_box.layout.display = "none"

    aoi_mode.observe(update_aoi_box, names="value")
    product_select.observe(update_sentinel_stat_box, names="value")

    # -----------------------------
    # Run button function
    # -----------------------------
    def on_run_clicked(button):
        with output:
            clear_output()

            try:
                print("Preparing AOI...")

                if aoi_mode.value == "Upload shapefile":
                    aoi = read_uploaded_shapefile(shapefile_upload)
                else:
                    aoi = create_aoi_from_latlon(
                        lat=lat_input.value,
                        lon=lon_input.value,
                        buffer_km=buffer_input.value
                    )

                selected_products = list(product_select.value)

                sentinel_stat = None
                if "Sentinel imagery" in selected_products:
                    sentinel_stat = sentinel_stat_select.value

                query = build_query(
                    aoi=aoi,
                    selected_products=selected_products,
                    start_year=start_year.value,
                    start_month=start_month.value,
                    end_year=end_year.value,
                    end_month=end_month.value,
                    sentinel_stat=sentinel_stat
                )

                print("AOI loaded successfully.")
                print(f"Selected products: {selected_products}")

                if sentinel_stat is not None:
                    print(f"Sentinel statistic selected: {sentinel_stat}")

                print("\nGenerated ODC query:")
                print(query)

                print("\nAOI bounds:")
                print(aoi.total_bounds)

                print("\nAOI preview:")
                display(aoi)

                # ---------------------------------------------------------
                # Connect your actual Open Data Cube loading code here
                # ---------------------------------------------------------
                # Example:
                #
                # import datacube
                # dc = datacube.Datacube(app="interactive_loader")
                #
                # for product in query["products"]:
                #     ds = dc.load(
                #         product=product,
                #         x=query["x"],
                #         y=query["y"],
                #         time=query["time"],
                #         output_crs=query["output_crs"],
                #         resolution=query["resolution"],
                #         group_by=query["group_by"]
                #     )
                #
                #     print(f"Loaded product: {product}")
                #     display(ds)
                #
                #     if product == get_product_mapping()["Sentinel imagery"]:
                #         ds_stat = apply_sentinel_statistic(ds, sentinel_stat)
                #         print(f"Applied Sentinel statistic: {sentinel_stat}")
                #         display(ds_stat)
                # ---------------------------------------------------------

            except Exception as e:
                print("Error:")
                print(e)

    def on_clear_clicked(button):
        with output:
            clear_output()

    run_button.on_click(on_run_clicked)
    clear_button.on_click(on_clear_clicked)

    # -----------------------------
    # Layout
    # -----------------------------
    left_panel = widgets.VBox([
        widgets.HTML("<h3>AOI Selection</h3>"),
        aoi_mode,
        dynamic_aoi_box,
        widgets.HTML("<hr>"),

        widgets.HTML("<h3>Product Selection</h3>"),
        product_select,
        sentinel_stat_box,
        widgets.HTML("<hr>"),

        widgets.HTML("<h3>Date Range</h3>"),
        start_year,
        start_month,
        end_year,
        end_month,
        widgets.HTML("<br>"),

        widgets.HBox([run_button, clear_button])
    ], layout=widgets.Layout(
        width="390px",
        border="1px solid lightgray",
        padding="12px"
    ))

    right_panel = widgets.VBox([
        widgets.HTML("<h3>Output</h3>"),
        output
    ], layout=widgets.Layout(
        width="760px",
        padding="12px"
    ))

    ui = widgets.HBox([left_panel, right_panel])

    update_aoi_box()
    update_sentinel_stat_box()

    display(ui)



# --- Surface mining screening workflow ---
# Surface_mining_screening_refactor.py

import os
from pathlib import Path
import datacube
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import display, Markdown

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from odc.geo.geom import Geometry
from deafrica_tools.bandindices import calculate_indices, dualpol_indices
from deafrica_tools.datahandling import load_ard
from deafrica_tools.plotting import map_shapefile, rgb
from deafrica_tools.spatial import xr_rasterize

from shapely.geometry import MultiPolygon, Polygon
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, disk

import rasterio
from rasterio.transform import from_bounds


S2_MINING_PRODUCTS = {"s2", "s2_semiannual", "s2_imagery"}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def pixel_area_km2_from_coords(ds: xr.Dataset | xr.DataArray) -> float:
    """
    Robust pixel area (km^2) from x/y coordinate spacing (handles negative resolution).
    Works for projected CRS data (e.g., EPSG:6933).
    """
    # x/y resolution in metres (absolute)
    dx = float(abs(ds.x[1] - ds.x[0]))
    dy = float(abs(ds.y[1] - ds.y[0]))
    return (dx * dy) / 1_000_000.0  # m^2 -> km^2


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def convert_3D_polygon_to_2D(poly_3D: Polygon) -> Polygon:
    exterior_2d = [(x, y) for x, y, *_ in poly_3D.exterior.coords]
    interiors_2d = [[(x, y) for x, y, *_ in ring.coords] for ring in poly_3D.interiors]
    return Polygon(exterior_2d, interiors_2d)


def convert_3D_geometry_to_2D(geom_3D: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    if geom_3D.geom_type == "Polygon":
        return convert_3D_polygon_to_2D(geom_3D)
    if geom_3D.geom_type == "MultiPolygon":
        return MultiPolygon([convert_3D_polygon_to_2D(p) for p in geom_3D.geoms])
    return geom_3D

def show_title_after(text: str):
    """Caption shown AFTER an output cell renders."""
    try:
        display(Markdown(f"*{text}*"))
    except Exception:
        print(text)

def show_df_after(df: pd.DataFrame, caption: str | None = None):
    display(df)
    if caption:
        show_title_after(caption)

def show_plot_after(caption: str | None = None):
    """Call this AFTER plt.show()."""
    if caption:
        show_title_after(caption)

def save_dataarray_geotiff(
    da: xr.DataArray,
    out_path: str | Path,
    nodata: float | int | None = None,
    dtype: str | None = None,
    compress: str = "deflate",
):
    """
    Save a 2D DataArray to GeoTIFF.
    Requires da.odc.geobox to exist (common in ODC/DE Africa).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if da.ndim != 2:
        raise ValueError(f"Expected 2D DataArray, got shape {da.shape} with dims {da.dims}")

    if not hasattr(da, "odc") or not hasattr(da.odc, "geobox"):
        # Fallback: construct transform from bounds (works if coords are regular)
        xmin, xmax = float(da.x.min()), float(da.x.max())
        ymin, ymax = float(da.y.min()), float(da.y.max())
        width, height = da.sizes["x"], da.sizes["y"]
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

        crs = None
        if hasattr(da, "geobox") and getattr(da.geobox, "crs", None) is not None:
            crs = da.geobox.crs
        # else: try attrs
        crs = crs or da.attrs.get("crs", None)
    else:
        gb = da.odc.geobox
        transform = gb.transform
        crs = gb.crs

    arr = da.values

    if dtype is None:
        dtype = str(arr.dtype)

    if nodata is None:
        # If float, use NaN; if int, set a safe sentinel
        if np.issubdtype(arr.dtype, np.floating):
            nodata = np.nan
        else:
            nodata = -9999

    # For float GeoTIFF, nodata cannot be NaN reliably in all stacks
    # so we convert NaNs to a finite nodata if needed.
    write_arr = arr
    write_nodata = nodata
    if np.issubdtype(np.dtype(dtype), np.floating):
        if np.isnan(nodata):
            # choose a conventional nodata
            write_nodata = -9999.0
        write_arr = np.where(np.isfinite(arr), arr, write_nodata).astype(dtype)
    else:
        # integer
        if np.issubdtype(arr.dtype, np.floating):
            write_arr = np.where(np.isfinite(arr), arr, write_nodata).astype(dtype)
        else:
            write_arr = arr.astype(dtype)

    profile = dict(
        driver="GTiff",
        height=write_arr.shape[0],
        width=write_arr.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=write_nodata,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(write_arr, 1)

def plot_rgb_and_mining_veg_loss_by_year(
    ds: xr.Dataset,
    vegetation_loss_bool: xr.DataArray,          # (time, y, x) bool
    veg_loss_in_buffer_mask: xr.DataArray,       # (y, x) 0/1 or bool
    product: str = "s2",
    out_png: str | Path | None = None,
    dpi: int = 300,
    max_years_in_legend: int | None = None,      # optional: cap legend size
):
    """
    Reproduces the classic 2-panel figure:
      (1) RGB plot from most recent composite image
      (2) Vegetation loss from possible mining by year (colored overlays)
    """

    index = "NDVI" if product in S2_MINING_PRODUCTS else "RVI"

    years = pd.to_datetime(vegetation_loss_bool.time.values).year
    if len(years) < 2:
        raise ValueError("Need at least 2 timesteps/years to plot vegetation loss by year.")

    # Background image: first year index (grey)
    background = ds[index].isel(time=0)

    # Most recent RGB index = last timestep
    last_i = len(years) - 1

    # Boolean mask per year: veg loss AND within mining buffer
    # Ensure boolean types
    loss_any = vegetation_loss_bool.fillna(False).astype(bool)
    
    buf = veg_loss_in_buffer_mask
    if buf.dtype != bool:
        buf = (buf == 1)
    buf = buf.fillna(False).astype(bool)
    
    # Broadcast buf (y,x) across time safely
    loss_in_mining = loss_any & buf

    # Build colors (repeatable and visible)
    # Use Tableau colors like the original style (orange/green/red etc)
    tableau = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    n = len(years)
    colors = [tableau[i % len(tableau)] for i in range(n)]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # ---- LEFT: RGB most recent ----
    ax0 = axes[0]
    if product in S2_MINING_PRODUCTS:
        rgb(ds, index=[last_i], ax=ax0)
    else:
        med_s1 = ds[["vv", "vh", "vh/vv"]].median()
        rgb(
            (ds[["vv", "vh", "vh/vv"]] / med_s1),
            bands=["vv", "vh", "vh/vv"],
            index=[last_i],
            ax=ax0,
        )
    ax0.set_title("RGB plot from most recent composite image")
    ax0.set_axis_off()

    # ---- RIGHT: background + yearly overlays ----
    ax1 = axes[1]
    background.plot.imshow(ax=ax1, cmap="Greys", add_colorbar=False)
    ax1.set_axis_off()

    # Legend items
    legend_patches = []
    legend_labels = []

    # optional: cap legend years (e.g. if 20+ years)
    year_indices = list(range(1, n))  # start from 1 (first year has no change)
    if max_years_in_legend is not None and len(year_indices) > max_years_in_legend:
        # keep most recent N years
        year_indices = year_indices[-max_years_in_legend:]

    # Plot each year overlay (like your screenshot)
    for i in year_indices:
        da = loss_in_mining.isel(time=i).where(loss_in_mining.isel(time=i) == True)
        # If no pixels that year, skip
        if int(da.fillna(False).sum().values) == 0:
            continue

        da.plot.imshow(
            ax=ax1,
            add_colorbar=False,
            cmap=ListedColormap([colors[i]]),
        )
        legend_patches.append(Patch(facecolor=colors[i]))
        legend_labels.append(str(int(years[i])))

    ax1.legend(legend_patches, legend_labels, loc="upper left")
    ax1.set_title(f"Vegetation Loss from Possible Mining from {int(years[0])} to {int(years[-1])}")

    # Save
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)

def save_time_stack_geotiffs(
    da_time: xr.DataArray,
    out_dir: str | Path,
    prefix: str,
    nodata: float | int | None = None,
    dtype: str | None = None,
):
    """
    Save a (time, y, x) DataArray as one GeoTIFF per timestep.
    """
    out_dir = _ensure_dir(out_dir)
    years = pd.to_datetime(da_time.time.values).year
    for i, y in enumerate(years):
        save_dataarray_geotiff(
            da_time.isel(time=i),
            out_dir / f"{prefix}_{int(y)}.tif",
            nodata=nodata,
            dtype=dtype,
        )


# -----------------------------------------------------------------------------
# Core pipeline functions
# -----------------------------------------------------------------------------
def load_vector_file(vector_file: str):
    extension = vector_file.split(".")[-1].lower()
    if extension == "kml":
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(vector_file, driver="KML")
    else:
        gdf = gpd.read_file(vector_file)

    gdf["geometry"] = gdf["geometry"].apply(convert_3D_geometry_to_2D)
    geom = Geometry(gdf.unary_union, gdf.crs)

    # optional interactive preview
    map_shapefile(gdf, attribute=gdf.columns[0], fillOpacity=0, weight=3)
    return gdf, geom


def _reduce_time_by_statistic(ds: xr.Dataset, statistic: str):
    """
    Reduce a time-series Dataset using the selected statistic.
    Used for Sentinel-2 imagery composites.
    """
    statistic = str(statistic or "Median").strip()

    if statistic == "Median":
        return ds.median(dim="time", skipna=True)
    if statistic == "Mean":
        return ds.mean(dim="time", skipna=True)
    if statistic == "Minimum":
        return ds.min(dim="time", skipna=True)
    if statistic == "Maximum":
        return ds.max(dim="time", skipna=True)
    if statistic == "Standard deviation":
        return ds.std(dim="time", skipna=True)
    if statistic == "Geomedian":
        try:
            from odc.algo import xr_geomedian
        except ImportError as exc:
            raise ImportError("Geomedian requires odc.algo.xr_geomedian.") from exc
        return xr_geomedian(ds)

    raise ValueError(f"Unsupported Sentinel statistic: {statistic}")


def _annual_composite_by_statistic(ds: xr.Dataset, statistic: str):
    """
    Create one composite per year from Sentinel-2 imagery using the selected statistic.
    This preserves a yearly time dimension for vegetation-loss screening.
    """
    composites = []

    for year_value, yearly_ds in ds.groupby("time.year"):
        comp = _reduce_time_by_statistic(yearly_ds, statistic)
        comp = comp.expand_dims(time=[pd.Timestamp(f"{int(year_value)}-12-31")])
        composites.append(comp)

    if len(composites) == 0:
        raise ValueError("No Sentinel-2 imagery was loaded for the selected date range.")

    return xr.concat(composites, dim="time")


def process_data(
    gdf,
    geom,
    start_year,
    end_year,
    product="s2_semiannual",
    output_crs="epsg:6933",
    sentinel_statistic="Median",
):
    """
    Load data for the AOI.

    Supported product values:
    - s2_semiannual: Sentinel-2 semi-annual GeoMAD product
    - s2_imagery: Sentinel-2 imagery loaded with load_ard and composited by statistic
    - s2: backward-compatible alias for annual Sentinel-2 GeoMAD
    - s1: backward-compatible Sentinel-1 RTC option
    """
    dc = datacube.Datacube(app="surface_mining")

    query = {"geopolygon": geom}

    if product == "s1":
        ds = load_ard(
            dc=dc,
            products=["s1_rtc"],
            time=(f"{start_year}", f"{end_year}"),
            measurements=["vv", "vh"],
            resolution=(-20, 20),
            output_crs=output_crs,
            group_by="solar_day",
            **query,
        )
        ds["vh/vv"] = ds.vh / ds.vv
        ds = dualpol_indices(ds, index="RVI")
        ds = _annual_composite_by_statistic(ds, sentinel_statistic)

    elif product == "s2_semiannual":
        ds = dc.load(
            product="gm_s2_semiannual",
            measurements=["red", "green", "blue", "nir"],
            time=(f"{start_year}", f"{end_year}"),
            resolution=(-10, 10),
            **query,
        )
        ds = calculate_indices(ds, ["NDVI"], satellite_mission="s2")

    elif product == "s2_imagery":
        ds = load_ard(
            dc=dc,
            products=["s2_l2a_c1"],
            measurements=["red", "green", "blue", "nir"],
            time=(f"{start_year}", f"{end_year}"),
            resolution=(-10, 10),
            output_crs=output_crs,
            group_by="solar_day",
            **query,
        )
        ds = calculate_indices(ds, ["NDVI"], satellite_mission="s2")
        ds = _annual_composite_by_statistic(ds, sentinel_statistic)

    elif product == "s2":
        ds = dc.load(
            product="gm_s2_annual",
            measurements=["red", "green", "blue", "nir"],
            time=(f"{start_year}", f"{end_year}"),
            resolution=(-10, 10),
            **query,
        )
        ds = calculate_indices(ds, ["NDVI"], satellite_mission="s2")

    else:
        raise ValueError("product must be 's2_semiannual', 's2_imagery', 's2', or 's1'")

    ds_wofs = dc.load(
        product="wofs_ls_summary_annual",
        time=(f"{start_year}", f"{end_year}"),
        resampling="nearest",
        like=ds.geobox,
    ).frequency

    # Rasterize AOI to dataset grid and mask all outputs
    mask = xr_rasterize(gdf, ds).astype(bool)

    ds = ds.where(mask)
    ds_wofs = ds_wofs.where(mask)

    # water appeared in >10% of observations in a given year
    water_bool = ds_wofs > 0.1
    water_frequency_sum = water_bool.sum("time").where(mask)

    return ds, water_frequency_sum, mask


def calculate_vegetation_loss(ds: xr.Dataset, product="s2", threshold=-0.15):
    """
    Compute year-to-year change and vegetation loss boolean:
      loss = (index[t] - index[t-1]) < threshold
    Returns:
      loss_bool(time,y,x), loss_sum(y,x), change(time,y,x)
    """
    index = "NDVI" if product in S2_MINING_PRODUCTS else "RVI"
    if index not in ds:
        raise KeyError(f"{index} not found in dataset. Available: {list(ds.data_vars)}")

    change = ds[index] - ds[index].shift(time=1)

    if threshold == "otsu":
        thr = threshold_otsu(np.nan_to_num(change.values, nan=0.0))
    else:
        thr = float(threshold)

    loss_bool = (change < thr)
    # keep NaNs outside AOI masked as NaN (not False)
    loss_bool = loss_bool.where(np.isfinite(ds[index]))

    # sum of years with loss (counts True as 1)
    loss_sum = loss_bool.fillna(False).astype(np.uint8).sum("time")

    return loss_bool, loss_sum, change, thr


def possible_mining_masks(
    vegetation_loss_sum: xr.DataArray,
    water_frequency_sum: xr.DataArray,
    ds: xr.Dataset,
    buffer_m: float = 90.0,
):
    """
    Efficient mining screening:
    - base_mining = (veg_loss_sum>0) & (water_frequency_sum>0 for at least one year)
    - buffered_mining = raster buffer via binary dilation (fast)
    - veg_loss_in_buffer = veg_loss_sum within buffered mining
    Returns:
      base_mining_mask, buffered_mining_mask, veg_loss_in_buffer_mask
    """
    # base mining candidate pixels
    base_mining = (vegetation_loss_sum > 0) & (water_frequency_sum > 0)
    base_mining = base_mining.fillna(False)

    # buffer in pixels
    # ds coords are in metres in projected CRS (e.g., EPSG:6933)
    res_m = float(abs(ds.x[1] - ds.x[0]))
    radius_px = int(np.ceil(buffer_m / res_m))
    if radius_px < 1:
        radius_px = 1

    buffered = binary_dilation(base_mining.values.astype(bool), footprint=disk(radius_px))
    buffered_mining = xr.DataArray(
        buffered.astype(np.uint8),
        coords=base_mining.coords,
        dims=base_mining.dims,
        name="buffered_mining",
    )

    veg_loss_in_buffer = (vegetation_loss_sum > 0) & (buffered_mining == 1)
    veg_loss_in_buffer = veg_loss_in_buffer.fillna(False).astype(np.uint8)

    base_mining_mask = base_mining.astype(np.uint8)
    return base_mining_mask, buffered_mining, veg_loss_in_buffer


def build_summary_table(
    ds: xr.Dataset,
    vegetation_loss_bool: xr.DataArray,
    veg_loss_in_buffer_mask: xr.DataArray,
    product="s2",
):
    """
    Builds a per-year summary table:
    - total AOI area (km2)
    - any vegetation loss area (km2, %)
    - vegetation loss within possible mining buffer area (km2, %)
    Returns DataFrame.
    """
    index = "NDVI" if product in S2_MINING_PRODUCTS else "RVI"
    background = ds[index].isel(time=0)

    pix_area = pixel_area_km2_from_coords(ds)
    total_area = int(np.count_nonzero(np.isfinite(background.values))) * pix_area

    years = pd.to_datetime(vegetation_loss_bool.time.values).year

    # any vegetation loss per year
    loss_any = vegetation_loss_bool.fillna(False).astype(np.uint8)
    loss_any_area = loss_any.sum(dim=["y", "x"]).values * pix_area

    # vegetation loss within buffer (per year):
    # veg_loss_in_buffer_mask is (y,x); apply it to yearly loss
    loss_in_buffer = (loss_any == 1) & (veg_loss_in_buffer_mask == 1)
    loss_in_buffer_area = loss_in_buffer.sum(dim=["y", "x"]).values * pix_area

    df = pd.DataFrame(
        {
            "year": years,
            "any_veg_loss_km2": loss_any_area,
            "any_veg_loss_%": (loss_any_area / total_area) * 100.0,
            "veg_loss_in_from_mining_km2": loss_in_buffer_area,
            "veg_loss_in_from_mining_%": (loss_in_buffer_area / total_area) * 100.0,
        }
    )
    meta = pd.DataFrame(
        {
            "metric": ["total_aoi_area_km2"],
            "value": [total_area],
        }
    )
    return df, meta


# -----------------------------------------------------------------------------
# Optional plotting helpers (no recompute)
# -----------------------------------------------------------------------------
def plot_possible_mining_map(
    ds: xr.Dataset,
    veg_loss_in_buffer_mask: xr.DataArray,
    product="s2",
    out_png: str | Path | None = None,
):
    index = "NDVI" if product in S2_MINING_PRODUCTS else "RVI"
    bg = ds[index].isel(time=0)

    plt.figure(figsize=(12, 12))
    bg.plot.imshow(cmap="Greys", add_colorbar=False)
    veg_loss_in_buffer_mask.where(veg_loss_in_buffer_mask == 1).plot.imshow(
        cmap=ListedColormap(["Gold"]), add_colorbar=False
    )
    plt.legend([Patch(facecolor="Gold")], ["Possible Mining Site"], loc="upper left")
    plt.title("Possible Mining Areas")
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()

    # plot_rgb_and_mining_veg_loss_by_year(
    #     ds=ds,
    #     vegetation_loss_bool=veg_loss_bool,
    #     veg_loss_in_buffer_mask=veg_loss_in_buffer_mask,
    #     product=product,
    #     out_png=out_dir / "RGB_and_VegLoss_From_Mining.png",
    #     dpi=dpi,
    #     max_years_in_legend=12,  # optional; remove if you want all years
    # )
    # show_plot_after("Saved: RGB_and_VegLoss_From_Mining.png (two-panel figure)")

def run_surface_mining_screening(
    vector_file: str,
    start_year: str,
    end_year: str,
    product: str = "s2_semiannual",
    sentinel_statistic: str | None = None,
    threshold: float | str = -0.15,
    buffer_m: float = 90.0,
    out_dir: str = "results",
    export_yearly_loss_geotiffs: bool = True,
    export_yearly_loss_pngs: bool = True,
    dpi: int = 300,
    col_wrap: int = 6,
    max_years_in_legend: int | None = 12,   # set None to show all
):
    """
    One-shot runner that:
      - loads AOI + imagery
      - computes vegetation loss + possible mining masks
      - exports CSV tables + GeoTIFF rasters
      - saves key figures as PNG at dpi=300 (including:
            - Possible mining map
            - Vegetation-loss time series
            - 2-panel figure: RGB (most recent) + Veg loss from mining by year
            - RGB per-year PNGs
        )
      - Notebook captions appear AFTER outputs (tables/plots)
    """

    # ----------------------------
    # Local helpers (caption AFTER)
    # ----------------------------
    def plot_and_save_veg_loss_timeseries(
        ds: xr.Dataset,
        vegetation_loss_bool: xr.DataArray,
        veg_loss_in_buffer_mask: xr.DataArray,
        out_png: str | Path,
    ):
        pix_area = pixel_area_km2_from_coords(ds)
        years = pd.to_datetime(vegetation_loss_bool.time.values).year

        loss_any = vegetation_loss_bool.fillna(False).astype(np.uint8)
        loss_any_area = loss_any.sum(dim=["y", "x"]).values * pix_area

        buf = (veg_loss_in_buffer_mask == 1) if veg_loss_in_buffer_mask.dtype != bool else veg_loss_in_buffer_mask
        loss_in_buffer = (loss_any == 1) & buf
        loss_in_buffer_area = loss_in_buffer.sum(dim=["y", "x"]).values * pix_area

        plt.figure(figsize=(11, 4))
        plt.plot(years, loss_any_area, marker="o", label="Any vegetation loss (km²)")
        plt.plot(years, loss_in_buffer_area, marker="^", label="Veg loss in mining buffer (km²)")
        plt.grid(True)
        plt.xlabel("Year")
        plt.ylabel("Area (km²)")
        plt.title("Annual Vegetation Loss")
        plt.legend()

        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.show()
        show_plot_after("Vegetation loss time series")

    def plot_possible_mining_map_local(
        ds: xr.Dataset,
        veg_loss_in_buffer_mask: xr.DataArray,
        out_png: str | Path,
    ):
        index = "NDVI" if product in S2_MINING_PRODUCTS else "RVI"
        bg = ds[index].isel(time=0)

        plt.figure(figsize=(12, 12))
        bg.plot.imshow(cmap="Greys", add_colorbar=False)
        veg_loss_in_buffer_mask.where(veg_loss_in_buffer_mask == 1).plot.imshow(
            cmap=ListedColormap(["Gold"]), add_colorbar=False
        )
        plt.legend([Patch(facecolor="Gold")], ["Possible Mining Site"], loc="upper left")
        plt.title("Possible Vegetation Loss Areas (Mining Screening)")

        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.show()
        show_plot_after("Possible mining map")

    def plot_rgb_and_mining_veg_loss_by_year(
        ds: xr.Dataset,
        vegetation_loss_bool: xr.DataArray,          # (time, y, x) bool-ish
        veg_loss_in_buffer_mask: xr.DataArray,       # (y, x) 0/1 or bool
        out_png: str | Path,
    ):
        """
        2-panel figure:
          (1) RGB plot from most recent composite image
          (2) Vegetation loss from possible mining by year (colored overlays)
        """
        index = "NDVI" if product in S2_MINING_PRODUCTS else "RVI"
    
        years = pd.to_datetime(vegetation_loss_bool.time.values).year
        if len(years) < 2:
            raise ValueError("Need at least 2 years to plot vegetation loss by year.")
    
        background = ds[index].isel(time=0)
        last_i = len(years) - 1
    
        # ---- FIX: boolean casting to avoid TypeError ----
        loss_any = vegetation_loss_bool.fillna(False).astype(bool)
    
        buf = veg_loss_in_buffer_mask
        if buf.dtype != bool:
            buf = (buf == 1)
        buf = buf.fillna(False).astype(bool)
    
        # Broadcast (y,x) buffer across (time,y,x)
        loss_in_mining = loss_any & buf
    
        # Colors
        tableau = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"
        ]
        n = len(years)
        colors = [tableau[i % len(tableau)] for i in range(n)]
    
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
        # LEFT: most recent RGB
        ax0 = axes[0]
        if product in S2_MINING_PRODUCTS:
            rgb(ds, index=[last_i], ax=ax0)
        else:
            med_s1 = ds[["vv", "vh", "vh/vv"]].median()
            rgb(
                (ds[["vv", "vh", "vh/vv"]] / med_s1),
                bands=["vv", "vh", "vh/vv"],
                index=[last_i],
                ax=ax0,
            )
        ax0.set_title("RGB plot from most recent composite image")
        ax0.set_axis_off()
    
        # RIGHT: background + yearly overlays
        ax1 = axes[1]
        background.plot.imshow(ax=ax1, cmap="Greys", add_colorbar=False)
        ax1.set_axis_off()
    
        legend_patches, legend_labels = [], []
    
        year_indices = list(range(1, n))  # skip first year
        if max_years_in_legend is not None and len(year_indices) > max_years_in_legend:
            year_indices = year_indices[-max_years_in_legend:]
    
        for i in year_indices:
            da = loss_in_mining.isel(time=i)
            # skip empty years
            if int(da.sum().values) == 0:
                continue
    
            da.where(da).plot.imshow(
                ax=ax1,
                add_colorbar=False,
                cmap=ListedColormap([colors[i]]),
            )
            legend_patches.append(Patch(facecolor=colors[i]))
            legend_labels.append(str(int(years[i])))
    
        ax1.legend(legend_patches, legend_labels, loc="upper left")
        ax1.set_title(f"Vegetation Loss from Possible Mining from {int(years[0])} to {int(years[-1])}")
    
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    
        plt.show()
        plt.close(fig)
        show_plot_after("Two-panel RGB + Vegetation loss from mining")


    def save_rgb_per_year_pngs(ds: xr.Dataset, out_folder: Path):
        out_folder.mkdir(parents=True, exist_ok=True)
        years = pd.to_datetime(ds.time.values).year

        for i, y in enumerate(years):
            fig, ax = plt.subplots(figsize=(7, 7))
            if product in S2_MINING_PRODUCTS:
                rgb(ds, index=[i], ax=ax)
                ax.set_title(f"RGB Composite - {int(y)}")
                fig.savefig(out_folder / f"rgb_{int(y)}.png", dpi=dpi, bbox_inches="tight")
            else:
                med_s1 = ds[["vv", "vh", "vh/vv"]].median()
                rgb(
                    (ds[["vv", "vh", "vh/vv"]] / med_s1),
                    bands=["vv", "vh", "vh/vv"],
                    index=[i],
                    ax=ax,
                )
                ax.set_title(f"S1 RGB (vv, vh, vh/vv) - {int(y)}")
                fig.savefig(out_folder / f"s1_rgb_{int(y)}.png", dpi=dpi, bbox_inches="tight")

            # plt.show()
            plt.close(fig)

        show_plot_after("Saved RGB composites per year (PNG, dpi=300)")

    # ----------------------------
    # Run pipeline
    # ----------------------------
    out_dir = _ensure_dir(out_dir)

    # 1) AOI
    gdf, geom = load_vector_file(vector_file)
    show_plot_after("Loaded AOI vector (interactive preview may have been displayed)")

    # 2) Data
    ds, water_frequency_sum, mask = process_data(
        gdf,
        geom,
        start_year,
        end_year,
        product=product,
        sentinel_statistic=sentinel_statistic or "Median",
    )
    show_plot_after("Loaded imagery and water summary; applied AOI mask")

    # 3) Vegetation loss
    veg_loss_bool, veg_loss_sum, change, thr = calculate_vegetation_loss(ds, product=product, threshold=threshold)
    show_title_after(f"Vegetation loss threshold used: {thr}")

    # 4) Mining masks
    base_mining_mask, buffered_mining_mask, veg_loss_in_buffer_mask = possible_mining_masks(
        veg_loss_sum, water_frequency_sum, ds, buffer_m=buffer_m
    )
    show_title_after("Computed possible mining masks")

    # 5) Tables
    df_yearly, df_meta = build_summary_table(ds, veg_loss_bool, veg_loss_in_buffer_mask, product=product)
    show_df_after(df_meta, "Summary metadata (also saved to CSV)")
    show_df_after(df_yearly, "Summary by year (also saved to CSV)")

    # 6) Export CSV
    df_yearly.to_csv(out_dir / "surface_mining_summary_by_year.csv", index=False)
    df_meta.to_csv(out_dir / "surface_mining_summary_meta.csv", index=False)
    show_title_after("Exported CSV tables to results folder")

    # 7) Export GeoTIFFs
    save_dataarray_geotiff(water_frequency_sum.astype(np.float32), out_dir / "water_frequency_sum.tif", nodata=-9999.0, dtype="float32")
    save_dataarray_geotiff(veg_loss_sum.astype(np.uint16), out_dir / "vegetation_loss_sum.tif", nodata=0, dtype="uint16")
    save_dataarray_geotiff(base_mining_mask.astype(np.uint8), out_dir / "possible_mining_base_mask.tif", nodata=0, dtype="uint8")
    save_dataarray_geotiff(buffered_mining_mask.astype(np.uint8), out_dir / "possible_mining_buffer_mask.tif", nodata=0, dtype="uint8")
    save_dataarray_geotiff(veg_loss_in_buffer_mask.astype(np.uint8), out_dir / "veg_loss_in_mining_buffer_mask.tif", nodata=0, dtype="uint8")
    show_title_after("Exported GeoTIFF outputs to results folder")

    if export_yearly_loss_geotiffs:
        save_time_stack_geotiffs(
            veg_loss_bool.fillna(False).astype(np.uint8),
            out_dir / "yearly_veg_loss_masks",
            prefix="veg_loss",
            nodata=0,
            dtype="uint8",
        )
        show_title_after("Exported yearly vegetation loss GeoTIFFs")

    # 8) Quick-look RGB (display + save per-year)
    if product in S2_MINING_PRODUCTS:
        rgb(ds, col="time", col_wrap=len(ds.time.values))
    else:
        med_s1 = ds[["vv", "vh", "vh/vv"]].median()
        rgb(ds[["vv", "vh", "vh/vv"]] / med_s1, bands=["vv", "vh", "vh/vv"], col="time", col_wrap=len(ds.time.values))
    show_plot_after("Quick-look RGB composites (displayed)")

    save_rgb_per_year_pngs(ds, out_dir / "rgb_yearly_pngs")

    # 9) Key figures
    plot_possible_mining_map_local(ds, veg_loss_in_buffer_mask, out_png=out_dir / "Possible_Mining.png")
    plot_and_save_veg_loss_timeseries(ds, veg_loss_bool, veg_loss_in_buffer_mask, out_png=out_dir / "veg_loss_timeseries.png")

    # *** This is the missing figure from your screenshot ***
    plot_rgb_and_mining_veg_loss_by_year(
        ds=ds,
        vegetation_loss_bool=veg_loss_bool,
        veg_loss_in_buffer_mask=veg_loss_in_buffer_mask,
        out_png=out_dir / "RGB_and_VegLoss_From_Mining.png",
    )


    show_title_after("Done ✅ All outputs are saved in the results folder.")
    return {
        "out_dir": str(out_dir),
        "summary_by_year": df_yearly,
        "summary_meta": df_meta,
        "threshold_used": thr,
    }




# =============================================================================
# Interactive UI wrapper for the Surface Mining Screening workflow
# =============================================================================

def _save_aoi_gdf_to_temp_file(aoi_gdf, prefix="interactive_aoi"):
    """
    Save a GeoDataFrame AOI to a temporary GeoJSON file and return the path.
    This lets the interactive UI feed uploaded or lat/lon AOIs into
    run_surface_mining_screening(), which expects a vector file path.
    """
    temp_dir = tempfile.mkdtemp(prefix=f"{prefix}_")
    out_path = Path(temp_dir) / "aoi.geojson"
    aoi_gdf.to_file(out_path, driver="GeoJSON")
    return str(out_path)


def display_surface_mining_screening_ui():
    """
    Display a horizontal Jupyter Notebook UI for the surface mining screening workflow.

    Product behaviour:
    - Semi-annual GeoMAD: uses year-only date controls.
    - Annual GeoMAD: uses year-only date controls.
    - Sentinel-2 imagery: shows month controls and statistic tools.
    - Sentinel-1: shows month controls and statistic tools.
    """

    # -----------------------------
    # AOI widgets
    # -----------------------------
    aoi_mode = widgets.ToggleButtons(
        options=["Upload shapefile", "Lat/Lon buffer"],
        description="AOI:",
        button_style="",
        layout=widgets.Layout(width="360px")
    )

    shapefile_upload = widgets.FileUpload(
        accept=".zip",
        multiple=False,
        description="Upload .zip",
        layout=widgets.Layout(width="220px")
    )

    lat_input = widgets.FloatText(
        value=5.6,
        description="Latitude:",
        layout=widgets.Layout(width="240px")
    )

    lon_input = widgets.FloatText(
        value=-0.2,
        description="Longitude:",
        layout=widgets.Layout(width="240px")
    )

    buffer_km_input = widgets.FloatText(
        value=5,
        description="AOI buffer km:",
        layout=widgets.Layout(width="240px")
    )

    shapefile_box = widgets.VBox([
        widgets.HTML("<b>Upload zipped shapefile</b>"),
        shapefile_upload,
        widgets.HTML("<small>Zip must contain .shp, .shx, .dbf, and .prj files.</small>")
    ])

    latlon_box = widgets.VBox([
        widgets.HTML("<b>Enter latitude, longitude, and AOI buffer</b>"),
        lat_input,
        lon_input,
        buffer_km_input
    ])

    dynamic_aoi_box = widgets.VBox([shapefile_box])

    # -----------------------------
    # Product widgets
    # -----------------------------
    mining_product = widgets.Dropdown(
        options={
            "Semi-annual GeoMAD": "s2_semiannual",
            "Sentinel-2 imagery": "s2_imagery",
        },
        value="s2_semiannual",
        description="Product:",
        layout=widgets.Layout(width="300px")
    )

    sentinel_stat_select = widgets.Dropdown(
        options=get_sentinel_stat_options(),
        value="Median",
        description="Statistic:",
        layout=widgets.Layout(width="300px")
    )

    sentinel_stat_box = widgets.VBox([
        widgets.HTML("<b>Sentinel-2 statistic</b>"),
        sentinel_stat_select,
        widgets.HTML("<small>Used to composite Sentinel-2 imagery by year.</small>")
    ])

    # -----------------------------
    # Date widgets
    # -----------------------------
    current_year = date.today().year
    years = list(range(2015, current_year + 2))
    months = list(range(1, 13))

    start_year = widgets.Dropdown(
        options=years,
        value=max(2018, current_year - 5),
        description="Start year:",
        layout=widgets.Layout(width="240px")
    )

    start_month = widgets.Dropdown(
        options=months,
        value=1,
        description="Start month:",
        layout=widgets.Layout(width="240px")
    )

    end_year = widgets.Dropdown(
        options=years,
        value=current_year,
        description="End year:",
        layout=widgets.Layout(width="240px")
    )

    end_month = widgets.Dropdown(
        options=months,
        value=12,
        description="End month:",
        layout=widgets.Layout(width="240px")
    )

    month_box = widgets.VBox([
        widgets.HTML("<b>Sentinel-2 month range</b>"),
        start_month,
        end_month,
        widgets.HTML("<small>Month controls appear only for Sentinel-2 imagery.</small>")
    ])

    # -----------------------------
    # Processing widgets
    # -----------------------------
    threshold_input = widgets.Text(
        value="-0.15",
        description="Threshold:",
        layout=widgets.Layout(width="260px")
    )

    mining_buffer_m = widgets.FloatText(
        value=90.0,
        description="Mining buffer m:",
        layout=widgets.Layout(width="260px")
    )

    out_dir_input = widgets.Text(
        value="results",
        description="Output folder:",
        layout=widgets.Layout(width="320px")
    )

    export_yearly_tifs = widgets.Checkbox(
        value=True,
        description="Export yearly vegetation-loss GeoTIFFs"
    )

    export_yearly_pngs = widgets.Checkbox(
        value=True,
        description="Export yearly PNGs"
    )

    run_button = widgets.Button(
        description="Run Screening",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="170px")
    )

    clear_button = widgets.Button(
        description="Clear Output",
        button_style="warning",
        icon="trash",
        layout=widgets.Layout(width="150px")
    )

    output = widgets.Output()

    # -----------------------------
    # UI behaviour
    # -----------------------------
    def update_aoi_box(change=None):
        if aoi_mode.value == "Upload shapefile":
            dynamic_aoi_box.children = [shapefile_box]
        else:
            dynamic_aoi_box.children = [latlon_box]

    def update_product_controls(change=None):
        supports_month_and_stat = mining_product.value in ["s2_imagery", "s1"]
        month_box.layout.display = "block" if supports_month_and_stat else "none"
        sentinel_stat_box.layout.display = "block" if supports_month_and_stat else "none"

    def parse_threshold(value):
        value = str(value).strip().lower()
        if value == "otsu":
            return "otsu"
        return float(value)

    def build_date_strings():
        sy = int(start_year.value)
        ey = int(end_year.value)

        if mining_product.value in ["s2_imagery", "s1"]:
            sm = int(start_month.value)
            em = int(end_month.value)
            last_day = calendar.monthrange(ey, em)[1]
            start_date = date(sy, sm, 1)
            end_date = date(ey, em, last_day)
        else:
            start_date = date(sy, 1, 1)
            end_date = date(ey, 12, 31)

        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date.")

        return start_date.isoformat(), end_date.isoformat()

    def on_run_clicked(button):
        with output:
            clear_output()
            try:
                print("Preparing AOI...")

                if aoi_mode.value == "Upload shapefile":
                    aoi_gdf = read_uploaded_shapefile(shapefile_upload)
                else:
                    aoi_gdf = create_aoi_from_latlon(
                        lat=lat_input.value,
                        lon=lon_input.value,
                        buffer_km=buffer_km_input.value
                    )

                vector_file = _save_aoi_gdf_to_temp_file(aoi_gdf)
                start_date, end_date = build_date_strings()
                threshold = parse_threshold(threshold_input.value)

                selected_product_label = mining_product.label
                sentinel_statistic = None
                if mining_product.value in ["s2_imagery", "s1"]:
                    sentinel_statistic = sentinel_stat_select.value

                print("AOI prepared successfully.")
                print(f"Temporary AOI file: {vector_file}")
                print(f"Product: {selected_product_label}")
                print(f"Date range: {start_date} to {end_date}")

                if sentinel_statistic is not None:
                    print(f"Sentinel-2 statistic: {sentinel_statistic}")

                print(f"Threshold: {threshold}")
                print(f"Mining buffer: {mining_buffer_m.value} m")
                print(f"Output folder: {out_dir_input.value}")
                print("\nRunning surface mining screening...")

                result = run_surface_mining_screening(
                    vector_file=vector_file,
                    start_year=start_date,
                    end_year=end_date,
                    product=mining_product.value,
                    sentinel_statistic=sentinel_statistic,
                    threshold=threshold,
                    buffer_m=float(mining_buffer_m.value),
                    out_dir=out_dir_input.value,
                    export_yearly_loss_geotiffs=bool(export_yearly_tifs.value),
                    export_yearly_loss_pngs=bool(export_yearly_pngs.value),
                )

                print("\nFinished.")
                display(result["summary_meta"])
                display(result["summary_by_year"])

            except Exception as e:
                print("Error:")
                print(e)

    def on_clear_clicked(button):
        with output:
            clear_output()

    aoi_mode.observe(update_aoi_box, names="value")
    mining_product.observe(update_product_controls, names="value")
    run_button.on_click(on_run_clicked)
    clear_button.on_click(on_clear_clicked)

    # -----------------------------
    # Horizontal layout
    # -----------------------------
    aoi_panel = widgets.VBox([
        widgets.HTML("<h3>AOI</h3>"),
        aoi_mode,
        dynamic_aoi_box,
    ], layout=widgets.Layout(
        width="360px",
        border="1px solid lightgray",
        padding="12px",
        margin="0 8px 0 0"
    ))

    product_panel = widgets.VBox([
        widgets.HTML("<h3>Product & Date</h3>"),
        mining_product,
        start_year,
        end_year,
        month_box,
        sentinel_stat_box,
    ], layout=widgets.Layout(
        width="340px",
        border="1px solid lightgray",
        padding="12px",
        margin="0 8px 0 0"
    ))

    processing_panel = widgets.VBox([
        widgets.HTML("<h3>Processing</h3>"),
        threshold_input,
        widgets.HTML("<small>Use a numeric threshold like -0.15 or type otsu.</small>"),
        mining_buffer_m,
        out_dir_input,
        export_yearly_tifs,
        export_yearly_pngs,
        widgets.HTML("<br>"),
        widgets.HBox([run_button, clear_button]),
    ], layout=widgets.Layout(
        width="360px",
        border="1px solid lightgray",
        padding="12px",
        margin="0 8px 0 0"
    ))

    controls_row = widgets.HBox([
        aoi_panel,
        product_panel,
        processing_panel,
    ], layout=widgets.Layout(
        width="100%",
        align_items="stretch"
    ))

    output_panel = widgets.VBox([
        widgets.HTML("<h3>Output</h3>"),
        output
    ], layout=widgets.Layout(
        width="100%",
        border="1px solid lightgray",
        padding="12px",
        margin="10px 0 0 0"
    ))

    update_aoi_box()
    update_aoi_shape_box()
    update_product_controls()

    display(widgets.VBox([controls_row, output_panel]))



# =============================================================================
# MEMORY-SAFE SURFACE MINING UI OVERRIDE
# =============================================================================
# The functions below intentionally override the earlier UI with a safer version.
# Main safeguards:
# - preview/estimate before loading data
# - default 30 m resolution instead of 10 m
# - optional plotting and GeoTIFF export
# - selectable area units: km² or hectares
# - AOI pixel limit before running
# - no automatic RGB grids over all years unless requested

import gc


def _load_vector_file_silent(vector_file: str):
    """Load vector file without displaying interactive map preview."""
    extension = vector_file.split(".")[-1].lower()
    if extension == "kml":
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(vector_file, driver="KML")
    else:
        gdf = gpd.read_file(vector_file)

    if gdf.empty:
        raise ValueError("AOI vector is empty.")
    if gdf.crs is None:
        raise ValueError("AOI vector has no CRS. Please define the projection first.")

    gdf["geometry"] = gdf["geometry"].apply(convert_3D_geometry_to_2D)
    geom = Geometry(gdf.unary_union, gdf.crs)
    return gdf, geom


def _estimate_aoi_pixels(aoi_gdf, resolution_m=30):
    """Estimate AOI area and pixel count for projected processing."""
    if aoi_gdf.crs is None:
        raise ValueError("AOI has no CRS.")
    gdf_proj = aoi_gdf.to_crs("EPSG:6933")
    area_m2 = float(gdf_proj.geometry.area.sum())
    pixel_count = area_m2 / float(resolution_m ** 2)
    return area_m2 / 1_000_000.0, int(pixel_count)


def _count_years_from_dates(start_date, end_date):
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    return int(end_dt.year - start_dt.year + 1)


def _convert_area_outputs(df_yearly, df_meta, area_unit="km2"):
    """
    Convert area columns in the summary tables to either km² or hectares.

    Parameters
    ----------
    df_yearly : pandas.DataFrame
        Summary table returned by build_summary_table(), with km² columns.
    df_meta : pandas.DataFrame
        Metadata table returned by build_summary_table(), with total_aoi_area_km2.
    area_unit : str
        Either 'km2' or 'hectares'.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Converted yearly and metadata tables.
    """
    unit = str(area_unit).strip().lower()
    if unit in ["km2", "km²", "square kilometres", "square kilometers"]:
        return df_yearly.copy(), df_meta.copy()
    if unit not in ["hectares", "ha", "hectare"]:
        raise ValueError("area_unit must be either 'km2' or 'hectares'.")

    yearly = df_yearly.copy()
    meta = df_meta.copy()

    rename_map = {
        "any_veg_loss_km2": "any_veg_loss_ha",
        "veg_loss_in_from_mining_km2": "veg_loss_in_from_mining_ha",
    }

    for src, dst in rename_map.items():
        if src in yearly.columns:
            yearly[dst] = yearly[src] * 100.0
            yearly = yearly.drop(columns=[src])

    if "metric" in meta.columns and "value" in meta.columns:
        mask = meta["metric"] == "total_aoi_area_km2"
        meta.loc[mask, "value"] = meta.loc[mask, "value"] * 100.0
        meta.loc[mask, "metric"] = "total_aoi_area_ha"

    return yearly, meta


def process_data_memory_safe(
    gdf,
    geom,
    start_date,
    end_date,
    product="s2_semiannual",
    output_crs="EPSG:6933",
    sentinel_statistic="Median",
    resolution_m=30,
    dask_chunks=None,
):
    """
    Memory-safer data loader for the surface-mining workflow.

    It uses a configurable output resolution and Dask chunking.

    Supported product values:
    - s2_semiannual: Sentinel-2 semi-annual GeoMAD
    - s2: Sentinel-2 annual GeoMAD
    - s2_imagery: Sentinel-2 imagery composited by selected statistic
    - s1: Sentinel-1 RTC composited annually by selected statistic after calculating RVI
    """
    dc = datacube.Datacube(app="surface_mining_safe")
    query = {"geopolygon": geom}

    if dask_chunks is None:
        dask_chunks = {"time": 1, "x": 1024, "y": 1024}

    resolution = (-float(resolution_m), float(resolution_m))

    if product == "s2_semiannual":
        ds = dc.load(
            product="gm_s2_semiannual",
            measurements=["red", "green", "blue", "nir"],
            time=(str(start_date), str(end_date)),
            resolution=resolution,
            output_crs=output_crs,
            dask_chunks=dask_chunks,
            **query,
        )
        ds = calculate_indices(ds, ["NDVI"], satellite_mission="s2")

    elif product == "s2":
        ds = dc.load(
            product="gm_s2_annual",
            measurements=["red", "green", "blue", "nir"],
            time=(str(start_date), str(end_date)),
            resolution=resolution,
            output_crs=output_crs,
            dask_chunks=dask_chunks,
            **query,
        )
        ds = calculate_indices(ds, ["NDVI"], satellite_mission="s2")

    elif product == "s2_imagery":
        ds = load_ard(
            dc=dc,
            products=["s2_l2a_c1"],
            measurements=["red", "green", "blue", "nir"],
            time=(str(start_date), str(end_date)),
            resolution=resolution,
            output_crs=output_crs,
            group_by="solar_day",
            dask_chunks=dask_chunks,
            **query,
        )
        ds = calculate_indices(ds, ["NDVI"], satellite_mission="s2")
        ds = _annual_composite_by_statistic(ds, sentinel_statistic)

    elif product == "s1":
        ds = load_ard(
            dc=dc,
            products=["s1_rtc"],
            measurements=["vv", "vh"],
            time=(str(start_date), str(end_date)),
            resolution=resolution,
            output_crs=output_crs,
            group_by="solar_day",
            dask_chunks=dask_chunks,
            **query,
        )
        ds["vh/vv"] = ds.vh / ds.vv
        ds = dualpol_indices(ds, index="RVI")
        ds = _annual_composite_by_statistic(ds, sentinel_statistic)

    else:
        raise ValueError("Safe UI supports 's2_semiannual', 's2', 's2_imagery', and 's1'.")

    ds_wofs = dc.load(
        product="wofs_ls_summary_annual",
        time=(str(start_date), str(end_date)),
        resampling="nearest",
        like=ds.geobox,
        dask_chunks=dask_chunks,
    ).frequency

    mask = xr_rasterize(gdf, ds).astype(bool)
    ds = ds.where(mask)
    ds_wofs = ds_wofs.where(mask)

    water_bool = ds_wofs > 0.1
    water_frequency_sum = water_bool.sum("time").where(mask)

    return ds, water_frequency_sum, mask



def _plot_vegetation_loss_timeseries_safe(
    ds,
    vegetation_loss_bool,
    veg_loss_in_buffer_mask,
    product="s2_semiannual",
    out_png=None,
    dpi=150,
):
    """
    Save/display a vegetation-loss time-series plot.
    This mirrors the original notebook time-series output but keeps it optional.
    """
    pix_area = pixel_area_km2_from_coords(ds)
    years = pd.to_datetime(vegetation_loss_bool.time.values).year

    loss_any = vegetation_loss_bool.fillna(False).astype(np.uint8)
    loss_any_area = loss_any.sum(dim=["y", "x"]).values * pix_area

    buf = veg_loss_in_buffer_mask
    if buf.dtype != bool:
        buf = (buf == 1)
    buf = buf.fillna(False).astype(bool)

    loss_in_buffer = (loss_any == 1) & buf
    loss_in_buffer_area = loss_in_buffer.sum(dim=["y", "x"]).values * pix_area

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(years, loss_any_area, marker="o", label="Any vegetation loss (km²)")
    ax.plot(years, loss_in_buffer_area, marker="^", label="Vegetation loss in mining buffer (km²)")
    ax.grid(True)
    ax.set_xlabel("Year")
    ax.set_ylabel("Area (km²)")
    ax.set_title("Annual vegetation loss")
    ax.legend()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def _normalise_plot_outputs(plot_outputs=None, make_plots=False):
    """Return a clean list of plot outputs requested by the UI."""
    if plot_outputs is None:
        return ["Possible mining map"] if make_plots else []
    if isinstance(plot_outputs, str):
        return [plot_outputs]
    return list(plot_outputs)

def run_surface_mining_screening_safe(
    vector_file: str,
    start_date: str,
    end_date: str,
    product: str = "s2_semiannual",
    sentinel_statistic: str | None = None,
    threshold: float | str = -0.15,
    buffer_m: float = 90.0,
    out_dir: str = "results_safe",
    resolution_m: int | float = 30,
    max_pixels: int = 5_000_000,
    area_unit: str = "km2",
    export_geotiffs: bool = False,
    export_yearly_loss_geotiffs: bool = False,
    make_plots: bool = False,
    plot_outputs: list | tuple | None = None,
    dpi: int = 150,
):
    """
    Memory-safe runner.

    By default, this does NOT create large RGB plot grids or GeoTIFF stacks.
    Enable plotting/export only after the preview estimate looks safe.
    """
    out_dir = _ensure_dir(out_dir)

    gdf, geom = _load_vector_file_silent(vector_file)
    area_km2, estimated_pixels = _estimate_aoi_pixels(gdf, resolution_m=resolution_m)
    n_years = _count_years_from_dates(start_date, end_date)

    if estimated_pixels > int(max_pixels):
        raise MemoryError(
            f"AOI is too large for the selected resolution. Estimated pixels: "
            f"{estimated_pixels:,}; limit: {int(max_pixels):,}. "
            f"Increase resolution_m, reduce AOI size, or increase max_pixels only if RAM allows."
        )

    if str(area_unit).strip().lower() in ["hectares", "ha", "hectare"]:
        print(f"AOI area: {area_km2 * 100.0:,.2f} ha")
    else:
        print(f"AOI area: {area_km2:,.2f} km²")
    print(f"Estimated pixels per layer: {estimated_pixels:,}")
    print(f"Date range years: {n_years}")
    print(f"Resolution: {resolution_m} m")

    ds, water_frequency_sum, mask = process_data_memory_safe(
        gdf=gdf,
        geom=geom,
        start_date=start_date,
        end_date=end_date,
        product=product,
        sentinel_statistic=sentinel_statistic or "Median",
        resolution_m=resolution_m,
    )
    print("Loaded imagery and water summary.")

    veg_loss_bool, veg_loss_sum, change, thr = calculate_vegetation_loss(
        ds, product=product, threshold=threshold
    )
    print(f"Vegetation-loss threshold used: {thr}")

    base_mining_mask, buffered_mining_mask, veg_loss_in_buffer_mask = possible_mining_masks(
        veg_loss_sum, water_frequency_sum, ds, buffer_m=buffer_m
    )
    print("Computed possible mining masks.")

    df_yearly, df_meta = build_summary_table(
        ds, veg_loss_bool, veg_loss_in_buffer_mask, product=product
    )
    df_yearly, df_meta = _convert_area_outputs(df_yearly, df_meta, area_unit=area_unit)

    df_yearly.to_csv(out_dir / "surface_mining_summary_by_year.csv", index=False)
    df_meta.to_csv(out_dir / "surface_mining_summary_meta.csv", index=False)
    print(f"Saved CSV outputs to: {out_dir}")

    if export_geotiffs:
        save_dataarray_geotiff(water_frequency_sum.astype(np.float32), out_dir / "water_frequency_sum.tif", nodata=-9999.0, dtype="float32")
        save_dataarray_geotiff(veg_loss_sum.astype(np.uint16), out_dir / "vegetation_loss_sum.tif", nodata=0, dtype="uint16")
        save_dataarray_geotiff(base_mining_mask.astype(np.uint8), out_dir / "possible_mining_base_mask.tif", nodata=0, dtype="uint8")
        save_dataarray_geotiff(buffered_mining_mask.astype(np.uint8), out_dir / "possible_mining_buffer_mask.tif", nodata=0, dtype="uint8")
        save_dataarray_geotiff(veg_loss_in_buffer_mask.astype(np.uint8), out_dir / "veg_loss_in_mining_buffer_mask.tif", nodata=0, dtype="uint8")
        print("Saved core GeoTIFF outputs.")

    if export_yearly_loss_geotiffs:
        save_time_stack_geotiffs(
            veg_loss_bool.fillna(False).astype(np.uint8),
            out_dir / "yearly_veg_loss_masks",
            prefix="veg_loss",
            nodata=0,
            dtype="uint8",
        )
        print("Saved yearly vegetation-loss GeoTIFFs.")

    selected_plot_outputs = _normalise_plot_outputs(plot_outputs=plot_outputs, make_plots=make_plots)

    if selected_plot_outputs:
        print("Creating selected plot outputs...")

        if "Possible mining map" in selected_plot_outputs:
            plot_possible_mining_map(
                ds=ds,
                veg_loss_in_buffer_mask=veg_loss_in_buffer_mask,
                product=product,
                out_png=out_dir / "Possible_Mining.png",
            )
            plt.close("all")
            gc.collect()

        if "Vegetation loss time series" in selected_plot_outputs:
            _plot_vegetation_loss_timeseries_safe(
                ds=ds,
                vegetation_loss_bool=veg_loss_bool,
                veg_loss_in_buffer_mask=veg_loss_in_buffer_mask,
                product=product,
                out_png=out_dir / "veg_loss_timeseries.png",
                dpi=dpi,
            )
            plt.close("all")
            gc.collect()

        if "Two-panel RGB + vegetation loss" in selected_plot_outputs:
            plot_rgb_and_mining_veg_loss_by_year(
                ds=ds,
                vegetation_loss_bool=veg_loss_bool,
                veg_loss_in_buffer_mask=veg_loss_in_buffer_mask,
                product=product,
                out_png=out_dir / "RGB_and_VegLoss_From_Mining.png",
                dpi=dpi,
                max_years_in_legend=12,
            )
            plt.close("all")
            gc.collect()

    result = {
        "out_dir": str(out_dir),
        "summary_by_year": df_yearly,
        "summary_meta": df_meta,
        "threshold_used": thr,
        "estimated_aoi_area_km2": area_km2,
        "estimated_aoi_area_ha": area_km2 * 100.0,
        "area_unit": area_unit,
        "estimated_pixels_per_layer": estimated_pixels,
    }

    # Reduce references before returning summaries.
    try:
        del ds, water_frequency_sum, mask, veg_loss_bool, veg_loss_sum, change
        del base_mining_mask, buffered_mining_mask, veg_loss_in_buffer_mask
    except Exception:
        pass
    gc.collect()

    return result


def display_surface_mining_screening_ui(
    *,
    read_uploaded_shapefile_fn=None,
    create_aoi_from_latlon_fn=None,
    save_aoi_gdf_to_temp_file_fn=None,
    estimate_aoi_pixels_fn=None,
    count_years_from_dates_fn=None,
    run_surface_mining_screening_safe_fn=None,
    get_sentinel_stat_options_fn=None,
):
    """
    Display a memory-safe, horizontal Jupyter Notebook UI for surface-mining screening.

    This function only handles the user interface. It does not contain the surface-
    mining processing workflow itself. The UI calls your existing backend functions
    when the user clicks Preview Size or Run Safe.

    Product behaviour
    -----------------
    - Semi-annual GeoMAD: uses year-only date controls.
    - Annual GeoMAD: uses year-only date controls.
    - Sentinel-2 imagery: shows month controls and statistic tools.
    - Sentinel-1: shows month controls and statistic tools.

    Area unit behaviour
    -------------------
    - User can select Square kilometres (km²) or Hectares (ha).
    - The selected value is passed to run_surface_mining_screening_safe().

    Memory-safety behaviour
    -----------------------
    - User must click Preview Size before Run Safe.
    - Preview estimates AOI area and pixels per layer.
    - Run Safe is blocked if Preview Size has not passed.

    Notes
    -----
    The function expects these helpers to exist in the global namespace unless
    they are supplied as keyword arguments:

    read_uploaded_shapefile
    create_aoi_from_latlon
    _save_aoi_gdf_to_temp_file
    _estimate_aoi_pixels
    _count_years_from_dates
    run_surface_mining_screening_safe
    get_sentinel_stat_options, optional
    """

    import calendar
    import gc
    from datetime import date

    import ipywidgets as widgets
    from IPython.display import display, clear_output

    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    g = globals()

    read_uploaded_shapefile_fn = read_uploaded_shapefile_fn or g.get("read_uploaded_shapefile")
    create_aoi_from_latlon_fn = create_aoi_from_latlon_fn or g.get("create_aoi_from_latlon")
    save_aoi_gdf_to_temp_file_fn = save_aoi_gdf_to_temp_file_fn or g.get("_save_aoi_gdf_to_temp_file")
    estimate_aoi_pixels_fn = estimate_aoi_pixels_fn or g.get("_estimate_aoi_pixels")
    count_years_from_dates_fn = count_years_from_dates_fn or g.get("_count_years_from_dates")
    run_surface_mining_screening_safe_fn = run_surface_mining_screening_safe_fn or g.get("run_surface_mining_screening_safe")
    get_sentinel_stat_options_fn = get_sentinel_stat_options_fn or g.get("get_sentinel_stat_options")

    required = {
        "read_uploaded_shapefile": read_uploaded_shapefile_fn,
        "create_aoi_from_latlon": create_aoi_from_latlon_fn,
        "_save_aoi_gdf_to_temp_file": save_aoi_gdf_to_temp_file_fn,
        "_estimate_aoi_pixels": estimate_aoi_pixels_fn,
        "_count_years_from_dates": count_years_from_dates_fn,
        "run_surface_mining_screening_safe": run_surface_mining_screening_safe_fn,
    }
    missing = [name for name, fn in required.items() if fn is None]
    if missing:
        raise NameError(
            "The UI-only function cannot run because these backend functions are missing: "
            + ", ".join(missing)
            + ". Import your processing module first, or pass these functions as keyword arguments."
        )

    if get_sentinel_stat_options_fn is not None:
        sentinel_stat_options = list(get_sentinel_stat_options_fn())
    else:
        sentinel_stat_options = [
            "Median",
            "Mean",
            "Minimum",
            "Maximum",
            "Standard deviation",
            "Geomedian",
        ]

    # ------------------------------------------------------------------
    # Help / instruction panel shown at the top of the UI
    # ------------------------------------------------------------------
    help_text = widgets.HTML("""
<h4>Surface Mining Screening Tool</h4>

<p>
This tool screens possible surface-mining areas by detecting vegetation loss over time
and comparing the loss with nearby water occurrence. Use <b>Preview Size</b> first to
check whether your AOI and date range are safe for the notebook memory.
</p>

<p>
<b>Step 1: Select Area of Interest</b><br>
Upload a zipped shapefile, or enter latitude/longitude and create a buffer. A circular
buffer is useful for a quick search around a point because it treats all directions equally.
A square or rectangular AOI is useful when you want a bounding-box style area, easier
comparison with map tiles, or a more conventional rectangular study area.
</p>

<p>
<b>Step 2: Choose Product</b><br>
<b>Semi-annual GeoMAD</b>: best first choice for stable, cloud-reduced Sentinel-2 composites.
It is usually lighter than loading many individual scenes.<br>
<b>Annual GeoMAD</b>: good for longer-term yearly vegetation change analysis with fewer time steps.<br>
<b>Sentinel-2 imagery</b>: use when you need a custom month range or custom statistic, but it is heavier because it loads direct imagery.<br>
<b>Sentinel-1</b>: radar option that can work in cloudy areas. It uses VV/VH radar information and RVI instead of NDVI.
</p>

<p>
<b>Step 3: Select Date Range</b><br>
Annual and semi-annual GeoMAD products use year-based controls. Sentinel-2 imagery and
Sentinel-1 also show month controls so you can restrict the image season.
</p>

<p>
<b>Step 4: Choose Image Statistic</b><br>
This appears for Sentinel-2 imagery and Sentinel-1. <b>Median</b> is the safest default and
is usually robust to outliers. <b>Mean</b> is useful for average conditions but can be affected
by outliers. <b>Minimum</b> and <b>Maximum</b> are useful for extremes. <b>Standard deviation</b>
highlights variability. <b>Geomedian</b> is robust for multi-band imagery but is more memory intensive.
</p>

<p>
<b>Step 5: Set Processing Options</b><br>
Choose vegetation-loss threshold, mining buffer distance, area unit, resolution, output folder,
and optional outputs. Start with a coarser resolution and small AOI before increasing detail.
</p>

<p>
<b>Step 6: Preview Before Running</b><br>
Use Preview Size to view the AOI on a map, estimate the AOI area and pixel count, and check whether the job is safe to run.
</p>

<p>
<b>Step 7: Run Processing</b><br>
Click Run Safe only after the preview status says the settings are OK.
</p>

<h4>Recommended first test</h4>

<p>
AOI buffer: 1–2 km, Product: Annual GeoMAD or Semi-annual GeoMAD, Resolution: 30 m or 60 m,
Date range: 2–3 years, Export GeoTIFFs: Off, Plot outputs: Possible mining map only.
</p>
""")

    # ------------------------------------------------------------------
    # AOI widgets
    # ------------------------------------------------------------------
    aoi_mode = widgets.ToggleButtons(
        options=["Upload shapefile", "Lat/Lon buffer"],
        description="AOI:",
        button_style="",
        layout=widgets.Layout(width="360px"),
    )

    shapefile_upload = widgets.FileUpload(
        accept=".zip",
        multiple=False,
        description="Upload .zip",
        layout=widgets.Layout(width="220px"),
    )

    lat_input = widgets.FloatText(
        value=5.6,
        description="Latitude:",
        layout=widgets.Layout(width="240px"),
    )
    lon_input = widgets.FloatText(
        value=-0.2,
        description="Longitude:",
        layout=widgets.Layout(width="240px"),
    )
    buffer_km_input = widgets.FloatText(
        value=2,
        description="AOI buffer km:",
        layout=widgets.Layout(width="240px"),
    )

    aoi_shape = widgets.ToggleButtons(
        options=["Circle buffer", "Square/rectangle"],
        value="Circle buffer",
        description="Shape:",
        button_style="",
        layout=widgets.Layout(width="360px"),
    )

    rect_width_km = widgets.FloatText(
        value=4,
        description="Width km:",
        layout=widgets.Layout(width="240px"),
    )

    rect_height_km = widgets.FloatText(
        value=4,
        description="Height km:",
        layout=widgets.Layout(width="240px"),
    )

    circle_shape_box = widgets.VBox([
        buffer_km_input,
        widgets.HTML("<small>Circle buffer is best for a quick search around a central point.</small>"),
    ])

    rectangle_shape_box = widgets.VBox([
        rect_width_km,
        rect_height_km,
        widgets.HTML("<small>Square/rectangle creates a bounding-box AOI around the point.</small>"),
    ])

    dynamic_shape_box = widgets.VBox([circle_shape_box])

    aoi_map_output = widgets.Output(layout=widgets.Layout(height="280px", overflow="auto"))

    preview_aoi_button = widgets.Button(
        description="Preview AOI Map",
        button_style="info",
        icon="map",
        layout=widgets.Layout(width="170px"),
    )

    shapefile_box = widgets.VBox([
        widgets.HTML("<b>Upload zipped shapefile</b>"),
        shapefile_upload,
        widgets.HTML("<small>Use a small AOI first. Large shapefiles can crash the notebook.</small>"),
    ])

    latlon_box = widgets.VBox([
        widgets.HTML("<b>Enter latitude, longitude, and AOI shape</b>"),
        lat_input,
        lon_input,
        aoi_shape,
        dynamic_shape_box,
    ])

    dynamic_aoi_box = widgets.VBox([shapefile_box])

    # ------------------------------------------------------------------
    # Product/date widgets
    # ------------------------------------------------------------------
    mining_product = widgets.Dropdown(
        options={
            "Semi-annual GeoMAD": "s2_semiannual",
            "Annual GeoMAD": "s2",
            "Sentinel-2 imagery": "s2_imagery",
            "Sentinel-1": "s1",
        },
        value="s2_semiannual",
        description="Product:",
        layout=widgets.Layout(width="300px"),
    )

    product_note = widgets.HTML(
        "<small><b>Semi-annual GeoMAD:</b> cloud-reduced composite; good first option for screening.</small>"
    )

    sentinel_stat_select = widgets.Dropdown(
        options=sentinel_stat_options,
        value="Median" if "Median" in sentinel_stat_options else sentinel_stat_options[0],
        description="Statistic:",
        layout=widgets.Layout(width="300px"),
    )

    statistic_title = widgets.HTML("<b>Image statistic</b>")
    statistic_note = widgets.HTML("<small>Median is safest. Geomedian is heavier.</small>")

    sentinel_stat_box = widgets.VBox([
        statistic_title,
        sentinel_stat_select,
        statistic_note,
    ])

    current_year = date.today().year
    years = list(range(2015, current_year + 2))
    months = list(range(1, 13))

    start_year = widgets.Dropdown(
        options=years,
        value=max(2018, current_year - 2),
        description="Start year:",
        layout=widgets.Layout(width="240px"),
    )
    start_month = widgets.Dropdown(
        options=months,
        value=1,
        description="Start month:",
        layout=widgets.Layout(width="240px"),
    )
    end_year = widgets.Dropdown(
        options=years,
        value=current_year,
        description="End year:",
        layout=widgets.Layout(width="240px"),
    )
    end_month = widgets.Dropdown(
        options=months,
        value=12,
        description="End month:",
        layout=widgets.Layout(width="240px"),
    )

    month_title = widgets.HTML("<b>Image month range</b>")
    month_note = widgets.HTML("<small>Month controls appear for Sentinel-2 imagery and Sentinel-1.</small>")

    month_box = widgets.VBox([
        month_title,
        start_month,
        end_month,
        month_note,
    ])

    # ------------------------------------------------------------------
    # Processing widgets
    # ------------------------------------------------------------------
    resolution_m = widgets.Dropdown(
        options=[10, 20, 30, 60, 100],
        value=30,
        description="Resolution m:",
        layout=widgets.Layout(width="260px"),
    )

    max_pixels = widgets.IntText(
        value=5_000_000,
        description="Max pixels:",
        layout=widgets.Layout(width="260px"),
    )

    area_unit = widgets.Dropdown(
        options={
            "Square kilometres (km²)": "km2",
            "Hectares (ha)": "hectares",
        },
        value="km2",
        description="Area unit:",
        layout=widgets.Layout(width="300px"),
    )

    threshold_input = widgets.Text(
        value="-0.15",
        description="Threshold:",
        layout=widgets.Layout(width="260px"),
    )
    mining_buffer_m = widgets.FloatText(
        value=90.0,
        description="Mining buffer m:",
        layout=widgets.Layout(width="260px"),
    )
    out_dir_input = widgets.Text(
        value="results_safe",
        description="Output folder:",
        layout=widgets.Layout(width="320px"),
    )

    export_core_tifs = widgets.Checkbox(value=False, description="Export core GeoTIFFs")
    export_yearly_tifs = widgets.Checkbox(value=False, description="Export yearly vegetation-loss GeoTIFFs")

    plot_outputs = widgets.SelectMultiple(
        options=[
            "Possible mining map",
            "Vegetation loss time series",
            "Two-panel RGB + vegetation loss",
        ],
        value=("Possible mining map",),
        description="Plots:",
        layout=widgets.Layout(width="360px", height="95px"),
    )

    preview_button = widgets.Button(
        description="Preview Size",
        button_style="info",
        icon="search",
        layout=widgets.Layout(width="150px"),
    )
    run_button = widgets.Button(
        description="Run Safe",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="140px"),
    )
    clear_button = widgets.Button(
        description="Clear",
        button_style="warning",
        icon="trash",
        layout=widgets.Layout(width="100px"),
    )

    output = widgets.Output()
    last_preview = {"ok": False}

    # ------------------------------------------------------------------
    # UI helper functions
    # ------------------------------------------------------------------
    def update_aoi_box(change=None):
        dynamic_aoi_box.children = [shapefile_box] if aoi_mode.value == "Upload shapefile" else [latlon_box]
        last_preview["ok"] = False

    def update_aoi_shape_box(change=None):
        if aoi_shape.value == "Circle buffer":
            dynamic_shape_box.children = [circle_shape_box]
        else:
            dynamic_shape_box.children = [rectangle_shape_box]
        last_preview["ok"] = False

    def create_rectangular_aoi_from_latlon(lat, lon, width_km, height_km, output_crs="EPSG:6933"):
        if width_km <= 0 or height_km <= 0:
            raise ValueError("Rectangle width and height must be greater than 0 km.")
        from shapely.geometry import Point, box
        import geopandas as gpd
        point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        projected = point_gdf.to_crs(output_crs)
        cx = float(projected.geometry.iloc[0].x)
        cy = float(projected.geometry.iloc[0].y)
        half_w = float(width_km) * 1000.0 / 2.0
        half_h = float(height_km) * 1000.0 / 2.0
        rect = box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        return gpd.GeoDataFrame(geometry=[rect], crs=output_crs).to_crs("EPSG:4326")

    def display_aoi_map(aoi_gdf):
        with aoi_map_output:
            clear_output()
            try:
                gdf4326 = aoi_gdf.to_crs("EPSG:4326")
                minx, miny, maxx, maxy = gdf4326.total_bounds
                center = ((miny + maxy) / 2.0, (minx + maxx) / 2.0)

                try:
                    from ipyleaflet import Map, GeoData, basemaps, LayersControl
                    m = Map(
                        center=center,
                        zoom=12,
                        basemap=basemaps.OpenStreetMap.Mapnik,
                        layout=widgets.Layout(width="100%", height="260px"),
                    )
                    geo_layer = GeoData(
                        geo_dataframe=gdf4326,
                        name="AOI",
                        style={"color": "red", "fillColor": "red", "opacity": 1, "weight": 2, "fillOpacity": 0.15},
                    )
                    m.add_layer(geo_layer)
                    m.add_control(LayersControl())
                    display(m)
                except Exception:
                    # Fallback if ipyleaflet is not installed.
                    try:
                        display(gdf4326.explore())
                    except Exception:
                        print("AOI bounds:", gdf4326.total_bounds)
                        display(gdf4326)
            except Exception as exc:
                print("Could not display AOI map:", exc)

    def update_product_controls(change=None):
        supports_month_and_stat = mining_product.value in ["s2_imagery", "s1"]

        if mining_product.value == "s2_semiannual":
            product_note.value = "<small><b>Semi-annual GeoMAD:</b> cloud-reduced Sentinel-2 composite; good first option for screening recent vegetation change.</small>"
        elif mining_product.value == "s2":
            product_note.value = "<small><b>Annual GeoMAD:</b> yearly cloud-reduced Sentinel-2 composite; useful for longer-term annual change with fewer timesteps.</small>"
        elif mining_product.value == "s2_imagery":
            product_note.value = "<small><b>Sentinel-2 imagery:</b> direct optical imagery. Use this when you need a custom month range or statistic, but expect heavier processing.</small>"
            statistic_title.value = "<b>Sentinel-2 statistic</b>"
            month_title.value = "<b>Sentinel-2 month range</b>"
            statistic_note.value = "<small>Median is safest. Mean shows average conditions. Min/Max show extremes. Standard deviation shows variability. Geomedian is robust but heavier.</small>"
            month_note.value = "<small>Use months to focus on the season of interest, for example the dry season or peak vegetation season.</small>"
        elif mining_product.value == "s1":
            product_note.value = "<small><b>Sentinel-1:</b> radar imagery for cloudy areas. This workflow uses VV/VH information and RVI rather than NDVI.</small>"
            statistic_title.value = "<b>Sentinel-1 statistic</b>"
            month_title.value = "<b>Sentinel-1 month range</b>"
            statistic_note.value = "<small>Median is safest for radar composites. Mean may be useful for average backscatter/RVI, while Min/Max highlight extremes.</small>"
            month_note.value = "<small>Use months to focus on a consistent seasonal window and reduce seasonal noise.</small>"

        month_box.layout.display = "block" if supports_month_and_stat else "none"
        sentinel_stat_box.layout.display = "block" if supports_month_and_stat else "none"
        last_preview["ok"] = False

    def invalidate_preview(change=None):
        last_preview["ok"] = False

    def parse_threshold(value):
        value = str(value).strip().lower()
        if value == "otsu":
            return "otsu"
        return float(value)

    def build_date_strings():
        sy = int(start_year.value)
        ey = int(end_year.value)

        if mining_product.value in ["s2_imagery", "s1"]:
            sm = int(start_month.value)
            em = int(end_month.value)
            last_day = calendar.monthrange(ey, em)[1]
            start_dt = date(sy, sm, 1)
            end_dt = date(ey, em, last_day)
        else:
            start_dt = date(sy, 1, 1)
            end_dt = date(ey, 12, 31)

        if start_dt > end_dt:
            raise ValueError("Start date must be before or equal to end date.")

        return start_dt.isoformat(), end_dt.isoformat()

    def prepare_aoi_file():
        if aoi_mode.value == "Upload shapefile":
            aoi_gdf = read_uploaded_shapefile_fn(shapefile_upload)
        else:
            if aoi_shape.value == "Circle buffer":
                aoi_gdf = create_aoi_from_latlon_fn(
                    lat=lat_input.value,
                    lon=lon_input.value,
                    buffer_km=buffer_km_input.value,
                )
            else:
                aoi_gdf = create_rectangular_aoi_from_latlon(
                    lat=lat_input.value,
                    lon=lon_input.value,
                    width_km=rect_width_km.value,
                    height_km=rect_height_km.value,
                )
        vector_file = save_aoi_gdf_to_temp_file_fn(aoi_gdf)
        return aoi_gdf, vector_file

    def on_preview_aoi_clicked(button):
        with output:
            clear_output()
            try:
                aoi_gdf, vector_file = prepare_aoi_file()
                display_aoi_map(aoi_gdf)
                print("AOI map preview updated.")
                print(f"Temporary AOI file: {vector_file}")
            except Exception as e:
                print("Error:")
                print(e)

    def on_preview_clicked(button):
        with output:
            clear_output()
            try:
                aoi_gdf, vector_file = prepare_aoi_file()
                display_aoi_map(aoi_gdf)
                start_dt, end_dt = build_date_strings()
                area_km2, pixel_estimate = estimate_aoi_pixels_fn(
                    aoi_gdf,
                    resolution_m=resolution_m.value,
                )
                n_years = count_years_from_dates_fn(start_dt, end_dt)
                if mining_product.value == "s1":
                    approx_layers = n_years * 3
                else:
                    approx_layers = n_years * 5
                approx_cells = int(pixel_estimate) * int(approx_layers)

                print("Preview estimate")
                print("----------------")
                print(f"AOI file: {vector_file}")
                print(f"Product: {mining_product.label}")
                print(f"Date range: {start_dt} to {end_dt}")

                if area_unit.value == "hectares":
                    print(f"AOI area: {area_km2 * 100.0:,.2f} ha")
                else:
                    print(f"AOI area: {area_km2:,.2f} km²")

                print(f"Estimated pixels per layer: {pixel_estimate:,}")
                print(f"Approximate analysis cells: {approx_cells:,}")
                print(f"Resolution: {resolution_m.value} m")
                print(f"Max pixel limit: {max_pixels.value:,}")

                if pixel_estimate > max_pixels.value:
                    print("\nStatus: NOT SAFE — reduce AOI size or increase resolution to 60/100 m.")
                    last_preview["ok"] = False
                else:
                    print("\nStatus: OK to try with current settings.")
                    last_preview["ok"] = True

                last_preview["vector_file"] = vector_file
                last_preview["start_dt"] = start_dt
                last_preview["end_dt"] = end_dt
                last_preview["product"] = mining_product.value
                last_preview["resolution_m"] = resolution_m.value
                last_preview["area_unit"] = area_unit.value

            except Exception as e:
                last_preview["ok"] = False
                print("Error:")
                print(e)

    def on_run_clicked(button):
        with output:
            try:
                if not last_preview.get("ok"):
                    print("Please click 'Preview Size' first and make sure the status is OK.")
                    return

                print("\nRunning memory-safe surface mining screening...")

                threshold = parse_threshold(threshold_input.value)
                sentinel_statistic = sentinel_stat_select.value if mining_product.value in ["s2_imagery", "s1"] else None

                result = run_surface_mining_screening_safe_fn(
                    vector_file=last_preview["vector_file"],
                    start_date=last_preview["start_dt"],
                    end_date=last_preview["end_dt"],
                    product=mining_product.value,
                    sentinel_statistic=sentinel_statistic,
                    threshold=threshold,
                    buffer_m=float(mining_buffer_m.value),
                    out_dir=out_dir_input.value,
                    resolution_m=float(resolution_m.value),
                    max_pixels=int(max_pixels.value),
                    area_unit=area_unit.value,
                    export_geotiffs=bool(export_core_tifs.value),
                    export_yearly_loss_geotiffs=bool(export_yearly_tifs.value),
                    make_plots=bool(plot_outputs.value),
                    plot_outputs=list(plot_outputs.value),
                )

                print("\nFinished.")

                if isinstance(result, dict):
                    if "summary_meta" in result:
                        display(result["summary_meta"])
                    if "summary_by_year" in result:
                        display(result["summary_by_year"])
                    if "out_dir" in result:
                        print(f"Outputs saved in: {result['out_dir']}")
                else:
                    display(result)

            except Exception as e:
                print("Error:")
                print(e)
            finally:
                if plt is not None:
                    plt.close("all")
                gc.collect()

    def on_clear_clicked(button):
        with output:
            clear_output()
        if plt is not None:
            plt.close("all")
        gc.collect()

    # Invalidate preview when key values change.
    for widget in [
        start_year,
        start_month,
        end_year,
        end_month,
        resolution_m,
        max_pixels,
        area_unit,
        threshold_input,
        mining_buffer_m,
        out_dir_input,
        sentinel_stat_select,
        buffer_km_input,
        rect_width_km,
        rect_height_km,
        lat_input,
        lon_input,
        aoi_shape,
        plot_outputs,
    ]:
        widget.observe(invalidate_preview, names="value")

    aoi_mode.observe(update_aoi_box, names="value")
    aoi_shape.observe(update_aoi_shape_box, names="value")
    mining_product.observe(update_product_controls, names="value")
    preview_aoi_button.on_click(on_preview_aoi_clicked)
    preview_button.on_click(on_preview_clicked)
    run_button.on_click(on_run_clicked)
    clear_button.on_click(on_clear_clicked)

    # ------------------------------------------------------------------
    # Horizontal layout
    # ------------------------------------------------------------------
    panel_style = dict(
        border="1px solid lightgray",
        padding="12px",
        margin="0 8px 0 0",
    )

    intro_panel = widgets.VBox([
        help_text,
    ], layout=widgets.Layout(
        width="100%",
        border="1px solid lightgray",
        padding="12px",
        margin="0 0 10px 0",
    ))

    aoi_panel = widgets.VBox([
        widgets.HTML("<h3>AOI</h3>"),
        aoi_mode,
        dynamic_aoi_box,
        preview_aoi_button,
        widgets.HTML("<b>AOI map preview</b>"),
        aoi_map_output,
    ], layout=widgets.Layout(width="390px", **panel_style))

    product_panel = widgets.VBox([
        widgets.HTML("<h3>Product & Date</h3>"),
        mining_product,
        product_note,
        start_year,
        end_year,
        month_box,
        sentinel_stat_box,
    ], layout=widgets.Layout(width="360px", **panel_style))

    processing_panel = widgets.VBox([
        widgets.HTML("<h3>Processing Safety</h3>"),
        resolution_m,
        max_pixels,
        area_unit,
        threshold_input,
        widgets.HTML("<small>Use a numeric threshold like -0.15 or type otsu.</small>"),
        mining_buffer_m,
        out_dir_input,
        export_core_tifs,
        export_yearly_tifs,
        widgets.HTML("<b>Optional plot outputs</b>"),
        plot_outputs,
        widgets.HTML("<small>The two-panel plot is useful but heavier. Start with only the possible mining map.</small>"),
        widgets.HTML("<br>"),
        widgets.HBox([preview_button, run_button, clear_button]),
    ], layout=widgets.Layout(width="390px", **panel_style))

    controls_row = widgets.HBox(
        [aoi_panel, product_panel, processing_panel],
        layout=widgets.Layout(width="100%", align_items="stretch"),
    )

    output_panel = widgets.VBox([
        widgets.HTML("<h3>Output</h3>"),
        output,
    ], layout=widgets.Layout(
        width="100%",
        border="1px solid lightgray",
        padding="12px",
        margin="10px 0 0 0",
    ))

    update_aoi_box()
    update_aoi_shape_box()
    update_product_controls()

    display(widgets.VBox([intro_panel, controls_row, output_panel]))