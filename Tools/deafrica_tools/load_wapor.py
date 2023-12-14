import calendar
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import rioxarray
import xarray as xr

pd.set_option("display.max_colwidth", None)

BASE_URL = "https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3/mapsets"


def get_WaPORv3_info(url: str, info: str | list[str]) -> pd.DataFrame:
    """
    Get information on WaPOR v3 data. WaPOR v3 variables are stored in `mapsets`,
    which in turn contain `rasters` that contain the data for a particular date or period.

    Parameters
    ----------
    url : str
        URL to get information from
    info : str | list[str]
        Attribute of a mapset

    Returns
    -------
    pd.DataFrame
        A table of the mapset attributes found.
    """
    if isinstance(info, str):
        info = [info]
    elif isinstance(info, list):
        info = [str(i) for i in info]
    else:
        raise TypeError("'info' must be a list or string.")

    data = {"links": [{"rel": "next", "href": url}]}

    output_dict = collections.defaultdict(list)
    while "next" in [x["rel"] for x in data["links"]]:
        url_ = [x["href"] for x in data["links"] if x["rel"] == "next"][0]
        response = requests.get(url_)
        response.raise_for_status()
        data = response.json()["response"]
        for item in data["items"]:
            if info == ["all"]:
                for key in list(item.keys()):
                    output_dict[key].append(item[key])
            else:
                for key in info:
                    output_dict[key].append(item[key])

    output_df = pd.DataFrame(output_dict)

    if "code" in output_df.columns:
        output_df.sort_values("code", inplace=True)
        output_df.reset_index(drop=True, inplace=True)
    return output_df


def get_all_WaPORv3_mapsets() -> pd.DataFrame:
    """
    List all the available WaPOR v3 mapset codes and descriptions.

    Returns
    -------
    pd.DataFrame
        A table of all the available WaPOR v3 mapset codes and descriptions.
    """
    output_df = get_WaPORv3_info(url=BASE_URL, info=["code", "caption"])
    output_df = output_df.rename(columns={"code": "Mapset Code", "caption": "Mapset Description"})

    level_dict = {"L1": "Global", "L2": "National", "L3": "Sub-national"}
    output_df["Level"] = output_df["Mapset Code"].apply(lambda x: level_dict[x.split("-")[0]])

    return output_df


def get_mapset_rasters(mapset_code: str) -> pd.DataFrame:
    """
    List all the available rasters for a mapset.

    Parameters
    ----------
    mapset_code : str
        Mapset code.

    Returns
    -------
    pd.DataFrame
        A table of the name and url of all the available mapset rasters.
    """
    mapset_url = f"{BASE_URL}/{mapset_code}/rasters"
    all_mapset_rasters = get_WaPORv3_info(url=mapset_url, info=["code", "downloadUrl"])
    return all_mapset_rasters


def get_dekad_start_dates(
    year: str | int, month: str | int
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Get the start dates of the 3 dekad intervals in a month.

    Parameters
    ----------
    year : str | int
        Year
    month : str | int
        Month

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]
        The start dates for the 3 dekads in a month.
    """

    year = int(year)
    month = int(month)

    # First day of the month
    start_date = datetime(year, month, 1)
    # Last day of the month.
    end_date = datetime(year, month, calendar.monthrange(year, month)[1])

    d1_start_date, d2_start_date, d3_start_date = pd.date_range(
        start=start_date, end=end_date, freq="10D", inclusive="left"
    )
    return d1_start_date, d2_start_date, d3_start_date


def get_dekad_label(date: str | datetime | pd.Timestamp) -> str:
    """
    Check the dekad of a date and return the dekad label.

    Parameters
    ----------
    date : str | datetime | pd.Timestamp
        Date to parse

    Returns
    -------
    str
        Dekad label.
    """
    # Get the year and month from the date.
    if isinstance(date, pd.Timestamp):
        timestamp = date
    else:
        timestamp = pd.Timestamp(date)
    year = timestamp.year
    month = timestamp.month

    d1_start_date, d2_start_date, d3_start_date = get_dekad_start_dates(year=year, month=month)

    if d1_start_date <= timestamp < d2_start_date:
        return f"{year}-{month:02d}-D1"
    elif d2_start_date <= timestamp < d3_start_date:
        return f"{year}-{month:02d}-D2"
    else:
        return f"{year}-{month:02d}-D3"


def get_month_label(date: str | datetime | pd.Timestamp) -> str:
    """
    Check the month of a date and return the month label.

    Parameters
    ----------
    date : str | datetime | pd.Timestamp
        Date to parse

    Returns
    -------
    str
        Month label.
    """
    # Get the year and month from the date.
    if isinstance(date, pd.Timestamp):
        timestamp = date
    else:
        timestamp = pd.Timestamp(date)
    year = timestamp.year
    month = timestamp.month

    return f"{year}-{month:02d}"


def get_year_label(date: str | datetime | pd.Timestamp) -> str:
    """
    Check the year of a date and return the year label.

    Parameters
    ----------
    date : str | datetime | pd.Timestamp
        Date to parse

    Returns
    -------
    str
        Year label.
    """
    # Get the year from the date.
    if isinstance(date, pd.Timestamp):
        timestamp = date
    else:
        timestamp = pd.Timestamp(date)
    year = timestamp.year

    return f"{year}"


def get_time_label(mapset_code: str, date: str | datetime | pd.Timestamp) -> str:
    """
    Get the label for a date  based on the mapset code.

    Parameters
    ----------
    mapset_code : str
        WaPOR v3 mapset code.
    date : str | datetime | pd.Timestamp
        Date to check.

    Returns
    -------
    str
        Time label
    """
    if mapset_code.split("-")[-1] == "A":
        time_label = get_year_label(date)
    elif mapset_code.split("-")[-1] == "M":
        time_label = get_month_label(date)
    elif mapset_code.split("-")[-1] == "D":
        time_label = get_dekad_label(date)

    return time_label


def get_time_from_url(url: str) -> np.datetime64:
    """
    Get the time to assign to a dataset from the WaPOR v3 url.

    Parameters
    ----------
    url : str
        URL to load

    Returns
    -------
    np.datetime64
        Time
    """
    mapset_code = url.split(".")[-3]
    time_label = url.split(".")[-2]

    # Get the time to assign the xarray.DataArray
    if mapset_code.split("-")[-1] == "A":
        date = datetime(
            int(time_label),
            1,
            1,
        )
        time = np.datetime64(date, "Y")
    elif mapset_code.split("-")[-1] == "M":
        year, month = time_label.split("-")
        date = datetime(int(year), int(month), 1)
        time = np.datetime64(date, "M")
    elif mapset_code.split("-")[-1] == "D":
        year, month, dekad_label = time_label.split("-")
        d1_start_date, d2_start_date, d3_start_date = get_dekad_start_dates(year=year, month=month)
        if dekad_label == "D1":
            time = np.datetime64(d1_start_date, "D")
        elif dekad_label == "D2":
            time = np.datetime64(d2_start_date, "D")
        else:
            time = np.datetime64(d3_start_date, "D")

    return time


def load_wapor(
    mapset_code: str,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    time_range: tuple[str, str],
) -> xr.Dataset:
    """
    Load a WaPOR v3 mapset.

    Parameters
    ----------
    mapset_code : str
    lat_range : tuple[float, float]
    lon_range : tuple[float, float]
    time_range : tuple[str, str]

    Returns
    -------
    xr.Dataset
        _description_
    """
    # Parse the time range.
    if isinstance(time_range, str):
        start_date = end_date = pd.to_datetime(time_range)
    elif isinstance(time_range, tuple):
        time_range_ = [pd.to_datetime(i) for i in time_range]
        start_date = min(time_range_)
        end_date = max(time_range_)
    else:
        raise TypeError(f"Expected time_range to be a tuple not {type(time_range)}")

    start_idx = get_time_label(mapset_code=mapset_code, date=start_date)
    end_idx = get_time_label(mapset_code=mapset_code, date=end_date)

    # Get coordinates from the longitude and latitude range.
    minx = min(lon_range)
    maxx = max(lon_range)
    miny = min(lat_range)
    maxy = max(lat_range)

    # Get a table of the rasters available for the mapset.
    df_mapset_rasters = get_mapset_rasters(mapset_code=mapset_code)
    df_mapset_rasters["dates"] = df_mapset_rasters["code"].apply(lambda x: x.split(".")[-1])
    df_mapset_rasters.set_index("dates", inplace=True)
    raster_urls = df_mapset_rasters.loc[start_idx:end_idx]["downloadUrl"].to_list()

    if raster_urls:
        da_list = []
        for url in raster_urls:
            da = (
                rioxarray.open_rasterio(url)
                .squeeze()
                .rio.clip_box(
                    minx=minx,
                    miny=miny,
                    maxx=maxx,
                    maxy=maxy,
                    crs="EPSG:4326",
                )
            )
            da = da.expand_dims(time=[get_time_from_url(url)])
            da = da.drop_vars("band")
            da_list.append(da)
        da_combined = xr.concat(da_list, dim="time")
        ds = da_combined.to_dataset(name=mapset_code)
        ds.attrs = da_combined.attrs
    else:
        print(f"No data available for the time range {time_range}")
        ds = xr.Dataset()

    return ds
