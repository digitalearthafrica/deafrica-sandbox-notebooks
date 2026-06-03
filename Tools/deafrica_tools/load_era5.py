"""
Functions to retrieve ERA5 gridded climate data.

Updated Apr 2026
"""

import numpy as np
import xarray as xr
import gcsfs
from odc.geo.xr import assign_crs
from dask.diagnostics import ProgressBar


ERA5_VARS = [
    "air_pressure_at_mean_sea_level",
    "air_temperature_at_2_metres",
    "eastward_wind_at_10_metres",
    "northward_wind_at_10_metres",
    "total_precipitation",
    "sea_surface_temperature",
    "surface_pressure",
]

ERA5_dict = {
    "air_pressure_at_mean_sea_level": "mean_sea_level_pressure",
    "air_temperature_at_2_metres": "2m_temperature",
    "eastward_wind_at_10_metres": "10m_u_component_of_wind",
    "northward_wind_at_10_metres": "10m_v_component_of_wind",
    "total_precipitation": "total_precipitation",
    "sea_surface_temperature": "sea_surface_temperature",
    "surface_pressure": "surface_pressure",
}

ARCO_BUCKET_PATH = (
    "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)


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


def _open_era5_zarr(chunks=None, consolidated=True):
    """
    Open public ARCO ERA5 Zarr using anonymous GCS access.
    """

    fs = gcsfs.GCSFileSystem(token="anon")

    store = fs.get_mapper(ARCO_BUCKET_PATH)

    try:
        ds = xr.open_zarr(
            store,
            consolidated=consolidated,
            chunks=chunks,
        )
    except PermissionError:
        print("Permission error with consolidated=True. Retrying with consolidated=False...")
        ds = xr.open_zarr(
            store,
            consolidated=False,
            chunks=chunks,
        )

    return ds


def load_era5(
    var,
    lat,
    lon,
    time,
    reduce_func="mean",
    resample="1D",
    chunks="auto",
    compute=False,
    show_progress=True,
):
    """
    Load ERA5 variable from public ARCO ERA5 Zarr.

    Parameters
    ----------
    var : str
        One of ERA5_VARS.

    lat : list/tuple
        Latitude range as (min_lat, max_lat).

    lon : list/tuple
        Longitude range as (min_lon, max_lon).

    time : str, np.datetime64, list or tuple
        Single date or date range.

    reduce_func : str or callable
        "mean", "sum", "min", "max", or a NumPy function.

    resample : str
        Resampling frequency, e.g. "1D", "1M".

    chunks : dict
        Dask chunks.

    compute : bool
        If True, loads result into memory.

    show_progress : bool
        If True, prints progress updates.

    Returns
    -------
    xarray.Dataset
    """

    if var not in ERA5_dict:
        raise ValueError(f"var must be one of {list(ERA5_dict)}. Got: {var}")

    if len(lat) != 2 or len(lon) != 2:
        raise ValueError("lat and lon must each have two values: (min, max)")

    date_from, date_to = _parse_time_range(time)

    lat_min, lat_max = min(lat), max(lat)
    lon_min, lon_max = _normalise_lon_bounds(lon)

    if show_progress:
        print("Opening ERA5 Zarr dataset...")
        print(f"Variable: {var}")
        print(f"Mapped ERA5 name: {ERA5_dict[var]}")
        print(f"Time: {date_from} to {date_to}")
        print(f"Latitude: {lat_min} to {lat_max}")
        print(f"Longitude: {lon_min} to {lon_max}")

    ds = _open_era5_zarr(chunks=chunks)

    arco_name = ERA5_dict[var]

    if arco_name not in ds.data_vars:
        raise KeyError(
            f"'{arco_name}' not found in ERA5 dataset. "
            f"Available variables include: {list(ds.data_vars)[:30]}"
        )

    da = ds[arco_name]

    if show_progress:
        print("Selecting time range...")

    da = da.sel(time=slice(str(date_from), str(date_to)))

    if da.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No ERA5 data found from {date_from} to {date_to}."
        )

    if show_progress:
        print("Normalising longitude coordinates...")

    if "longitude" in da.coords:
        da = da.assign_coords(
            longitude=(((da.longitude + 180) % 360) - 180)
        ).sortby("longitude")

    if show_progress:
        print("Selecting AOI...")

    # --- SNAP TO ERA5 GRID ---
    snapped = da.sel(
        latitude=list((lat_min, lat_max)),
        longitude=list((lon_min, lon_max)),
        method="nearest"
    )
    
    lat_range = slice(snapped.latitude.max().values, snapped.latitude.min().values)
    lon_range = slice(snapped.longitude.min().values, snapped.longitude.max().values)
    
    da = da.sel(
        latitude=lat_range,
        longitude=lon_range,
    )

    if da.sizes.get("latitude", 0) == 0 or da.sizes.get("longitude", 0) == 0:
        raise ValueError(
            "No ERA5 pixels found for this AOI. "
            "Try expanding the latitude/longitude bounds slightly."
        )

    if show_progress:
        print(f"Subset size: {dict(da.sizes)}")
        print(f"Resampling to {resample} using {reduce_func}...")

    if isinstance(reduce_func, str):
        reduce_func = reduce_func.lower()

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
                "reduce_func must be 'mean', 'sum', 'min', 'max', or a callable."
            )
    else:
        da = da.resample(time=resample).reduce(reduce_func)

    out = da.to_dataset(name=var)

    out = out.rename(
        {
            "latitude": "lat",
            "longitude": "lon",
        }
    )

    out = assign_crs(out, "EPSG:4326")

    if compute:
        if show_progress:
            print("Computing data now...")

        if show_progress:
            with ProgressBar():
                out = out.compute()
        else:
            out = out.compute()

    if show_progress:
        print("ERA5 loading complete.")

    return out