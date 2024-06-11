"""
Functions to retrieve Global Root-zone moisture Analysis & Forecasting System (GRAFS)
data.
"""
import numpy as np
import xarray as xr


def load_soil_moisture(
    lat: tuple[float | int] | list[float | int],
    lon: tuple[float | int] | list[float | int],
    time: tuple[str, str],
    product: str = "surface",
    grid: str = "nearest",
) -> xr.Dataset:
    """
    Load the Global Root-zone moisture Analysis & Forecasting System (GRAFS)
    data.

    Parameters
    ----------
    lat : tuple[float  |  int] | list[float  |  int]
        Latitude range for query.
    lon : tuple[float  |  int] | list[float  |  int]
        Longitude range for query.
    time : tuple[str, str]
        Time range to load data for.
    product : str, optional
        Band to load, must be "surface" or "root-zone", by default "surface"
    grid : str, optional
        Defines how the area of interest is defined from the lon and lat , by default "nearest"

    Returns
    -------
    xr.Dataset
        GRAFs surface or root-zone data found matching the query.
    """
    product_baseurl = "https://dapds00.nci.org.au/thredds/dodsC/ub8/global/GRAFS/"
    assert product in ["surface", "rootzone"], "product parameter must be surface or root-zone"
    # lat, lon grid
    if grid == "nearest":
        # select lat/lon range from data; snap to nearest grid
        lat_range, lon_range = None, None
    else:
        # define a grid that covers the entire area of interest
        lat_range = np.arange(
            np.max(np.ceil(np.array(lat) * 10.0 + 0.5) / 10.0 - 0.05),
            np.min(np.floor(np.array(lat) * 10.0 - 0.5) / 10.0 + 0.05) - 0.05,
            -0.1,
        )
        lon_range = np.arange(
            np.min(np.floor(np.array(lon) * 10.0 - 0.5) / 10.0 + 0.05),
            np.max(np.ceil(np.array(lon) * 10.0 + 0.5) / 10.0 - 0.05) + 0.05,
            0.1,
        )
    # split time window into years
    day_range = np.array(time).astype("M8[D]")
    year_range = np.array(time).astype("M8[Y]")
    if product == "surface":
        product_name = "GRAFS_TopSoilRelativeWetness_"
    else:
        product_name = "GRAFS_RootzoneSoilWaterIndex_"
    datasets = []
    for year in np.arange(year_range[0], year_range[1] + 1, np.timedelta64(1, "Y")):
        start = np.max([day_range[0], year.astype("M8[D]")])
        end = np.min([day_range[1], (year + 1).astype("M8[D]") - 1])
        product_url = product_baseurl + product_name + "%s.nc" % str(year)
        print(product_url)
        # data is loaded lazily through OPeNDAP
        ds = xr.open_dataset(product_url)
        if lat_range is None:
            # select lat/lon range from data if not specified; snap to nearest grid
            test = ds.sel(lat=list(lat), lon=list(lon), method="nearest")
            lat_range = slice(test.lat.values[0], test.lat.values[1])
            lon_range = slice(test.lon.values[0], test.lon.values[1])
        # slice before return
        ds = ds.sel(lat=lat_range, lon=lon_range, time=slice(start, end)).compute()
        datasets.append(ds)
    return xr.merge(datasets)
