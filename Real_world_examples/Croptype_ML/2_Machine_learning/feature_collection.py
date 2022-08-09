import sys
import datacube
import numpy as np
import xarray as xr
from deafrica_tools.datahandling import load_ard
from deafrica_tools.bandindices import calculate_indices
from odc.algo import xr_geomedian


def geomedian_with_indices_wrapper(ds):
    """Apply xr_geomedian function and calculate a series of indices"""

    indices = ["NDVI", "LAI", "SAVI", "MSAVI", "MNDWI"]
    satellite_mission = "s2"

    ds_geomedian = xr_geomedian(ds)

    ds_geomedian = calculate_indices(
        ds_geomedian,
        index=indices,
        drop=False,
        satellite_mission=satellite_mission,
    )

    return ds_geomedian


def indices_wrapper(ds):
    """Calculate list of indices"""

    indices = ["NDVI", "LAI", "SAVI", "MSAVI", "MNDWI"]
    satellite_mission = "s2"

    ds = calculate_indices(
        ds,
        index=indices,
        drop=False,
        satellite_mission=satellite_mission,
    )

    return ds


def median_wrapper(ds):
    """Calculate median"""

    ds = ds.median(dim="time")

    return ds


def mean_wrapper(ds):
    """calculate median"""

    ds = ds.mean(dim="time")

    return ds


def apply_function_over_custom_times(ds, func, func_name, time_ranges):
    """Apply generic function over an xarray dataset"""

    output_list = []

    for timelabel, timeslice in time_ranges.items():

        if isinstance(timeslice, slice):
            ds_timeslice = ds.sel(time=timeslice)
        else:
            ds_timeslice = ds.sel(time=timeslice, method="nearest")

        ds_modified = func(ds_timeslice)

        rename_dict = {
            key: f"{key}_{func_name}_{timelabel}" for key in list(ds_modified.keys())
        }

        ds_modified = ds_modified.rename(name_dict=rename_dict)

        if "time" in list(ds_modified.coords):
            ds_modified = ds_modified.reset_coords().drop_vars(["time", "spatial_ref"])

        output_list.append(ds_modified)

    return output_list


# Define functions to load features
def feature_layers(query):
    """Compute feature layers according to datacube query"""
    
#     query = static_query.copy() # include to make sure original query isn't modified

    # Connnect to datacube
    dc = datacube.Datacube(app="crop_type_ml")

#     # Check query for required time ranges and remove them
#     if all(
#         [
#             key in query.keys()
#             for key in [
#                 "time_ranges",
#                 "annual_geomedian_times",
#                 "semiannual_geomedian_times",
#             ]
#         ]
#     ):
#         pass
#     else:
#         print(
#             "Query missing at least one of time_ranges, annual_geomedian_times, or semiannual_geomedian_times"
#         )
#         sys.exit(1)

#     # ----------------- STORE TIME RANGES FOR CUSTOM QUERIES -----------------
#     # This removes these items from the query so it can be used for loads
#     time_ranges = query.pop("time_ranges")
#     annual_geomedian_times = query.pop("annual_geomedian_times")
#     semiannual_geomedian_times = query.pop("semiannual_geomedian_times")

    # ----------------- HARDCODE TIME RANGES FOR CUSTOM QUERIES -----------------
    # This means the function can be used without modifying the datacube query
    
    time_ranges = {
        "Q3_2021": slice("2021-08-01", "2022-10-31"),
        "Q4_2021": slice("2021-11-01", "2022-01-31"),
        "Q1_2022": slice("2022-02-01", "2022-04-30"),
    }
    
    # !!! FOR ZAMBIA, S1 DATA IS MISSING FOR HALF THE COUNTRY IN 2022 !!!
    s1_time_ranges = {
        "Q3_2021": slice("2021-08-01", "2022-10-31"),
        "Q4_2021": slice("2021-11-01", "2022-01-31"),
    }
    
    annual_geomedian_times = {
        "annual_2021": "2021-01-01",
    }
    semiannual_geomedian_times = {
        "semiannual_2021_01": "2021-01-01",
        "semiannual_2021_06": "2021-06-01",
    }

    # ----------------- DEFINE MEASUREMENTS TO USE FOR EACH PRODUCT -----------------

    s2_measurements = [
        "blue",
        "green",
        "red",
        "nir",
        "swir_1",
        "swir_2",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
    ]

    s2_geomad_measurements = s2_measurements + ["smad", "emad", "bcmad"]

    s1_measurements = ["vv", "vh"]

    fc_measurements = ["bs", "pv", "npv", "ue"]

    rainfall_measurements = ["rainfall"]

    slope_measurements = ["slope"]

    # ----------------- S2 CUSTOM GEOMEDIANS -----------------
    # These are designed to take the geomedian for every range in time_ranges
    # This is controlled through the input query

    ds = load_ard(
        dc=dc,
        products=["s2_l2a"],
        measurements=s2_measurements,
        group_by="solar_day",
        verbose=False,
        **query,
    )

    # Apply geomedian over time ranges and calculate band indices
    s2_geomad_list = apply_function_over_custom_times(
        ds, geomedian_with_indices_wrapper, "s2", time_ranges
    )

    # ----------------- S2 ANNUAL GEOMEDIAN -----------------

    # Update query to use annual_geomedian_times
    ds_annual_geomad_query = query.copy()
    query_times = list(annual_geomedian_times.values())
    ds_annual_geomad_query.update({"time": (query_times[0], query_times[-1])})

    # load s2 annual geomedian
    ds_s2_geomad = dc.load(
        product="gm_s2_annual",
        measurements=s2_geomad_measurements,
        **ds_annual_geomad_query,
    )

    # Calculate band indices
    s2_annual_list = apply_function_over_custom_times(
        ds_s2_geomad, indices_wrapper, "s2", annual_geomedian_times
    )

    # ----------------- S2 SEMIANNUAL GEOMEDIAN -----------------

    # Update query to use semiannual_geomedian_times
    ds_semiannual_geomad_query = query.copy()
    query_times = list(semiannual_geomedian_times.values())
    ds_semiannual_geomad_query.update({"time": (query_times[0], query_times[-1])})

    # load s2 semiannual geomedian
    ds_s2_semiannual_geomad = dc.load(
        product="gm_s2_semiannual",
        measurements=s2_geomad_measurements,
        **ds_semiannual_geomad_query,
    )

    # Calculate band indices
    s2_semiannual_list = apply_function_over_custom_times(
        ds_s2_semiannual_geomad, indices_wrapper, "s2", semiannual_geomedian_times
    )

    # ----------------- S1 CUSTOM GEOMEDIANS -----------------

    # Update query to suit Sentinel 1
    s1_query = query.copy()
    s1_query.update({"sat_orbit_state": "ascending"})

    # Load s1
    s1_ds = load_ard(
        dc=dc,
        products=["s1_rtc"],
        measurements=s1_measurements,
        group_by="solar_day",
        verbose=False,
        **s1_query,
    )

    # Apply geomedian
    s1_geomad_list = apply_function_over_custom_times(
        s1_ds, xr_geomedian, "s1_xrgm", s1_time_ranges
    )

    # -------- LANDSAT BIMONTHLY FRACTIONAL COVER -----------

    # Update query to suit fractional cover
    fc_query = query.copy()
    fc_query.update({"resampling": "bilinear", "measurements": fc_measurements})

    # load fractional cover
    ds_fc = dc.load(product="fc_ls", collection_category="T1", **fc_query)

    # Apply median
    fc_median_list = apply_function_over_custom_times(
        ds_fc, median_wrapper, "median", time_ranges
    )

    # -------- CHIRPS MONTHLY RAINFALL -----------

    # Update query to suit CHIRPS rainfall
    rainfall_query = query.copy()
    rainfall_query.update(
        {"resampling": "bilinear", "measurements": rainfall_measurements}
    )

    # Load rainfall and update no data values
    ds_rainfall = dc.load(product="rainfall_chirps_monthly", **rainfall_query)

    rainfall_nodata = -9999.0
    ds_rainfall = ds_rainfall.where(
        ds_rainfall.rainfall != rainfall_nodata, other=np.nan
    )

    # Apply mean
    rainfall_mean_list = apply_function_over_custom_times(
        ds_rainfall, mean_wrapper, "mean", time_ranges
    )

    # -------- DEM SLOPE -----------
    slope_query = query.copy()
    slope_query.update(
        {
            "resampling": "bilinear",
            "measurements": slope_measurements,
            "time": "2000-01-01",
        }
    )

    # Load slope data and update no data values and coordinates
    ds_slope = dc.load(product="dem_srtm_deriv", **slope_query)

    slope_nodata = -9999.0
    ds_slope = ds_slope.where(ds_slope != slope_nodata, np.nan)

    ds_slope = ds_slope.squeeze("time")#.reset_coords("time", drop=True)

    # ----------------- FINAL MERGED XARRAY -----------------

    # Create a list to keep all items for final merge
    ds_list = []
    ds_list.extend(s2_geomad_list)
    ds_list.extend(s2_annual_list)
    ds_list.extend(s2_semiannual_list)
    ds_list.extend(s1_geomad_list)
    ds_list.extend(fc_median_list)
    ds_list.extend(rainfall_mean_list)
    ds_list.append(ds_slope)

    ds_final = xr.merge(ds_list)

    return ds_final
