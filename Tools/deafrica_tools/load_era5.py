"""
Functions to retrieve ERA5 gridded climate data.

Updated Apr 2025 to directly access Zarr format data in PDS

"""

import numpy as np
import xarray as xr
from odc.geo.xr import assign_crs
from dask.diagnostics import ProgressBar

ERA5_VARS = [
    "air_pressure_at_mean_sea_level",
    "air_temperature_at_2_metres",
    "eastward_wind_at_10_metres",
    "northward_wind_at_10_metres",
    "total_precipitation_6hr",
    "total_precipitation_12hr",
    "total_precipitation_24hr",
    "sea_surface_temperature",
    "surface_pressure",
]

ERA5_dict = {
    "air_pressure_at_mean_sea_level": "mean_sea_level_pressure",
    "air_temperature_at_2_metres": "2m_temperature",
    "eastward_wind_at_10_metres": "10m_u_component_of_wind",
    "northward_wind_at_10_metres": "10m_v_component_of_wind",
    "total_precipitation_6hr": "total_precipitation_6hr",
    "total_precipitation_12hr": "total_precipitation_12hr",
    "total_precipitation_24hr": "total_precipitation_24hr",
    "sea_surface_temperature": "sea_surface_temperature",
    "surface_pressure": "surface_pressure",
}

ARCO_FULL_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def _parse_time_range(time):
    if isinstance(time, (list, tuple)):
        date_from = np.datetime64(min(time)).astype("datetime64[D]")
        date_to = np.datetime64(max(time)).astype("datetime64[D]")
    elif isinstance(time, (str, np.datetime64)):
        date_from = np.datetime64(time).astype("datetime64[D]")
        date_to = date_from
    else:
        raise ValueError("time must be str, np.datetime64, or (start, end)")

    return date_from, date_to


def _normalise_lon_bounds(lon):
    lon_min, lon_max = min(lon), max(lon)

    lon_min = ((lon_min + 180) % 360) - 180
    lon_max = ((lon_max + 180) % 360) - 180

    return lon_min, lon_max


def load_era5(
    var,
    lat,
    lon,
    time,
    reduce_func="mean",
    resample="1D",
    chunks=None,
    compute=False,
    show_progress=True,
):
    """
    Load an ERA5 variable from ARCO-ERA5 Zarr, subset to AOI/time, and resample.

    Parameters
    ----------
    var : str
        Friendly ERA5 variable name. Must be one of ERA5_VARS.

    lat : tuple/list
        Latitude bounds as (min_lat, max_lat).

    lon : tuple/list
        Longitude bounds as (min_lon, max_lon).

    time : str, np.datetime64, or tuple/list
        Single date or date range, e.g. "2024-01-01" or ("2024-01-01", "2024-01-31").

    reduce_func : str or callable
        Resampling reducer. Options: "mean", "sum", "min", "max", or callable such as np.mean.

    resample : str
        Resampling frequency, e.g. "1D", "1M", "6H".

    chunks : dict or None
        Optional Dask chunks. Example:
        {"time": 24, "latitude": 200, "longitude": 200}

    compute : bool
        If True, loads the result into memory immediately.

    show_progress : bool
        If True, prints progress messages and shows Dask ProgressBar when compute=True.

    Returns
    -------
    xarray.Dataset
        ERA5 subset with dimensions renamed to lat/lon and CRS assigned as EPSG:4326.
    """

    if var not in ERA5_dict:
        raise ValueError(f"var must be one of {list(ERA5_dict)}. Got: {var}")

    if len(lat) != 2 or len(lon) != 2:
        raise ValueError("lat and lon must each contain exactly two values: (min, max)")

    date_from, date_to = _parse_time_range(time)

    lat_min, lat_max = min(lat), max(lat)
    lon_min, lon_max = _normalise_lon_bounds(lon)

    if show_progress:
        print(f"Opening ERA5 Zarr dataset...")
        print(f"Variable: {var}")
        print(f"Time range: {date_from} to {date_to}")
        print(f"Lat range: {lat_min} to {lat_max}")
        print(f"Lon range: {lon_min} to {lon_max}")

    ds = xr.open_zarr(
        ARCO_FULL_URL,
        consolidated=True,
        storage_options={"token": "anon"},
        chunks=chunks,
    )

    arco_name = ERA5_dict[var]

    if arco_name not in ds.data_vars:
        raise KeyError(
            f"'{arco_name}' not found in ARCO dataset. "
            f"Available variables include: {list(ds.data_vars)[:30]}"
        )

    da = ds[arco_name]

    if show_progress:
        print("Selecting time range...")

    da = da.sel(time=slice(str(date_from), str(date_to)))

    if da.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No ERA5 data found for time range {date_from} to {date_to}."
        )

    if show_progress:
        print("Normalising longitude coordinates...")

    if "longitude" in da.coords:
        da = da.assign_coords(
            longitude=(((da.longitude + 180) % 360) - 180)
        ).sortby("longitude")

    if show_progress:
        print("Selecting spatial subset...")

    # ERA5 latitude is usually descending, so slice must go max to min
    da = da.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
    )

    if da.sizes.get("latitude", 0) == 0 or da.sizes.get("longitude", 0) == 0:
        raise ValueError(
            "No ERA5 grid cells found within the requested lat/lon bounds. "
            "Try expanding your AOI slightly."
        )

    if show_progress:
        print(f"Subset size: {dict(da.sizes)}")
        print(f"Resampling using: {reduce_func}")

    if isinstance(reduce_func, str):
        if reduce_func == "mean":
            da = da.resample(time=resample).mean()
        elif reduce_func == "sum":
            da = da.resample(time=resample).sum()
        elif reduce_func == "min":
            da = da.resample(time=resample).min()
        elif reduce_func == "max":
            da = da.resample(time=resample).max()
        else:
            raise ValueError(
                "reduce_func must be one of: 'mean', 'sum', 'min', 'max', or a callable"
            )
    else:
        da = da.resample(time=resample).reduce(reduce_func)

    out = da.to_dataset(name=var).rename(
        {
            "latitude": "lat",
            "longitude": "lon",
        }
    )

    out = assign_crs(out, "EPSG:4326")

    if compute:
        if show_progress:
            print("Computing/loading ERA5 data into memory...")

        if show_progress:
            with ProgressBar():
                out = out.compute()
        else:
            out = out.compute()

    if show_progress:
        print("ERA5 loading complete.")

    return out