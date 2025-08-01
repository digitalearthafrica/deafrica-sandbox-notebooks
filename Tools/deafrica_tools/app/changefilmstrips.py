"""
Loading and interacting with data in the change filmstrips notebook,
inside the Real_world_examples folder.
"""

import warnings

import datacube
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from datacube.utils.geometry import CRS, assign_crs
from eo_tides.eo import tag_tides
from ipyleaflet import basemap_to_tiles, basemaps
from odc.algo import geomedian_with_mads
from odc.ui import select_on_a_map

from deafrica_tools.dask import create_local_dask_cluster
from deafrica_tools.datahandling import load_ard, mostcommon_crs


def run_filmstrip_app(
    output_name: str,
    time_range: tuple,
    time_step: dict[str, int],
    tide_range: tuple[float] = (0.0, 1.0),
    resolution: tuple[int] = (-30, 30),
    max_cloud: float = 0.5,
    ls7_slc_off: bool = False,
    size_limit: int = 10000,
):
    """
    An interactive app that allows the user to select a region from a
    map, then load Digital Earth Africa Landsat data and combine it
    using the geometric median ("geomedian") statistic to reveal the
    median or 'typical' appearance of the landscape for a series of
    time periods.

    The results for each time period are combined into a 'filmstrip'
    plot which visualises how the landscape has changed in appearance
    across time, with a 'change heatmap' panel highlighting potential
    areas of greatest change.

    For coastal applications, the analysis can be customised to select
    only satellite images obtained during a specific tidal range
    (e.g. low, average or high tide).

    Last modified: April 2020

    Parameters
    ----------
    output_name : str
        A name that will be used to name the output filmstrip plot file.
    time_range : tuple
        A tuple giving the date range to analyse
        (e.g. `time_range = ('1988-01-01', '2017-12-31')`).
    time_step : dict
        This parameter sets the length of the time periods to compare
        (e.g. `time_step = {'years': 5}` will generate one filmstrip
        plot for every five years of data; `time_step = {'months': 18}`
        will generate one plot for each 18 month period etc. Time
        periods are counted from the first value given in `time_range`.
    tide_range : tuple, optional
        An optional parameter that can be used to generate filmstrip
        plots based on specific ocean tide conditions. This can be
        valuable for analysing change consistently along the coast.
        For example, `tide_range = (0.0, 0.2)` will select only
        satellite images acquired at the lowest 20% of tides;
        `tide_range = (0.8, 1.0)` will select images from the highest
        20% of tides. The default is `tide_range = (0.0, 1.0)` which
        will select all images regardless of tide.
    resolution : tuple, optional
        The spatial resolution to load data. The default is
        `resolution = (-30, 30)`, which will load data at 30 m pixel
        resolution. Increasing this (e.g. to `resolution = (-100, 100)`)
        can be useful for loading large spatial extents.
    max_cloud : float, optional
        This parameter can be used to exclude satellite images with
        excessive cloud. The default is `0.5`, which will keep all images
        with less than 50% cloud.
    ls7_slc_off : bool, optional
        An optional boolean indicating whether to include data from
        after the Landsat 7 SLC failure (i.e. SLC-off). Defaults to
        False, which removes all Landsat 7 observations > May 31 2003.
    size_limit : int, optional
        An optional integer (in hectares) specifying the size limit
        for the data query. Queries larger than this size will receive
        a warning that the data query is too large (and may
        therefore result in memory errors).


    Returns
    -------
    ds_geomedian : xarray Dataset
        An xarray dataset containing geomedian composites for each
        timestep in the analysis.

    """

    ########################
    # Select and load data #
    ########################

    # Define centre_coords as a global variable
    global centre_coords

    # Test if centre_coords is in the global namespace;
    # use default value if it isn't
    if "centre_coords" not in globals():
        centre_coords = (6.587292, 1.532833)

    # Plot interactive map to select area
    basemap = basemap_to_tiles(basemaps.Esri.WorldImagery)
    geopolygon = select_on_a_map(height="600px", layers=(basemap,), center=centre_coords, zoom=14)

    # Set centre coords based on most recent selection to re-focus
    # subsequent data selections
    centre_coords = geopolygon.centroid.points[0][::-1]

    # Test size of selected area
    msq_per_hectare = 10000
    area = geopolygon.to_crs(crs=CRS("epsg:6933")).area / msq_per_hectare
    radius = np.round(np.sqrt(size_limit), 1)  # noqa F841
    if area > size_limit:
        print(
            f"Warning: Your selected area is {area:.00f} hectares. "
            f"Please select an area of less than {size_limit} hectares."
            f"\nTo select a smaller area, re-run the cell "
            f"above and draw a new polygon."
        )

    else:
        print("Starting analysis...")

        # Connect to datacube database
        dc = datacube.Datacube(app="Change_filmstrips")

        # Configure local dask cluster
        client = create_local_dask_cluster(return_client=True)

        # Obtain native CRS
        crs = mostcommon_crs(
            dc=dc, product="ls8_sr", query={"time": "2014", "geopolygon": geopolygon}
        )

        # Create query based on time range, area selected, custom params
        query = {
            "time": time_range,
            "geopolygon": geopolygon,
            "output_crs": crs,
            "resolution": resolution,
            "dask_chunks": {"x": 3000, "y": 3000},
            "align": (resolution[1] / 2.0, resolution[1] / 2.0),
        }

        # Load data from all three Landsats
        warnings.filterwarnings("ignore")
        ds = load_ard(
            dc=dc,
            measurements=["red", "green", "blue"],
            products=["ls5_sr", "ls7_sr", "ls8_sr"],
            min_gooddata=max_cloud,
            ls7_slc_off=ls7_slc_off,
            **query,
        )

        # Optionally calculate tides for each timestep in the satellite
        # dataset and drop any observations outside this range
        if tide_range != (0.0, 1.0):
            ds = tag_tides(
                ds=ds,
                model="FES2014",
                directory="/var/share/tide_models",
            )
            min_tide, max_tide = ds.tide_height.quantile(tide_range).values
            ds = ds.sel(time=(ds.tide_height >= min_tide) & (ds.tide_height <= max_tide))
            ds = ds.drop("tide_height")
            print(
                f"    Keeping {len(ds.time)} observations with tides "
                f"between {min_tide:.2f} and {max_tide:.2f} m"
            )

        # Create time step ranges to generate filmstrips from
        bins_dt = pd.date_range(
            start=time_range[0], end=time_range[1], freq=pd.DateOffset(**time_step)
        )

        # Bin all satellite observations by timestep. If some observations
        # fall outside the upper bin, label these with the highest bin
        labels = bins_dt.astype("str")
        time_steps = (
            pd.cut(ds.time.values, bins_dt, labels=labels[:-1])
            .add_categories(labels[-1])
            .fillna(labels[-1])
        )

        time_steps_var = xr.DataArray(time_steps, [("time", ds.time.values)], name="timestep")

        # Resample data temporally into time steps, and compute geomedians
        ds_geomedian = ds.groupby(time_steps_var).apply(
            lambda ds_subset: geomedian_with_mads(
                ds_subset, compute_mads=False, compute_count=False
            )
        )

        print(
            "\nGenerating geomedian composites and plotting "
            "filmstrips... (click the Dashboard link above for status)"
        )
        ds_geomedian = ds_geomedian.compute()

        # Reset CRS that is lost during geomedian compositing
        ds_geomedian = assign_crs(ds_geomedian, crs=ds.geobox.crs)

        ############
        # Plotting #
        ############

        # Convert to array and extract vmin/vmax
        output_array = ds_geomedian[["red", "green", "blue"]].to_array()
        percentiles = output_array.quantile(q=(0.02, 0.98)).values

        # Create the plot with one subplot more than timesteps in the
        # dataset. Figure width is set based on the number of subplots
        # and aspect ratio
        n_obs = output_array.sizes["timestep"]
        ratio = output_array.sizes["x"] / output_array.sizes["y"]
        fig, axes = plt.subplots(1, n_obs + 1, figsize=(5 * ratio * (n_obs + 1), 5))
        fig.subplots_adjust(wspace=0.05, hspace=0.05)

        # Add timesteps to the plot, set aspect to equal to preserve shape
        for i, ax_i in enumerate(axes.flatten()[:n_obs]):
            output_array.isel(timestep=i).plot.imshow(
                ax=ax_i, vmin=percentiles[0], vmax=percentiles[1]
            )
            ax_i.get_xaxis().set_visible(False)
            ax_i.get_yaxis().set_visible(False)
            ax_i.set_aspect("equal")

        # Add change heatmap panel to final subplot. Heatmap is computed
        # by first taking the log of the array (so change in dark areas
        # can be identified), then computing standard deviation between
        # all timesteps
        (
            np.log(output_array)
            .std(dim=["timestep"])
            .mean(dim="variable")
            .plot.imshow(ax=axes.flatten()[-1], robust=True, cmap="magma", add_colorbar=False)
        )
        axes.flatten()[-1].get_xaxis().set_visible(False)
        axes.flatten()[-1].get_yaxis().set_visible(False)
        axes.flatten()[-1].set_aspect("equal")
        axes.flatten()[-1].set_title("Change heatmap")

        # Export to file
        date_string = "_".join(time_range)
        ts_v = list(time_step.values())[0]
        ts_k = list(time_step.keys())[0]
        fig.savefig(
            f"filmstrip_{output_name}_{date_string}_{ts_v}{ts_k}.png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        # close dask client
        client.shutdown()

        return ds_geomedian
