##Resample the data timeseries into **dekadal** (10-day) timesteps.

import calendar
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr

from xarray.core.groupby import DataArrayGroupBy, DatasetGroupBy

def get_dekad(date: np.datetime64) -> np.datetime64:
    """
    Get the start date of the dekad that a date belongs to.
    Every month has three dekads, such that the first two dekads
    have 10 days (i.e., 1-10, 11-20), and the third is comprised of the
    remaining days of the month.

    Parameters
    ----------
    date : np.datetime64
        Date to check.

    Returns
    -------
    np.datetime64
        Start date of the dekad.
    """
    timestamp = pd.Timestamp(date)
    year = timestamp.year
    month = timestamp.month

    first_day = datetime(year, month, 1)
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])

    d1_start_date, d2_start_date, d3_start_date = pd.date_range(
        start=first_day, end=last_day, freq="10D", inclusive="left"
    )

    if d1_start_date <= timestamp < d2_start_date:
        return np.datetime64(d1_start_date, "ns")
    elif d2_start_date <= timestamp < d3_start_date:
        return np.datetime64(d2_start_date, "ns")
    else:
        return np.datetime64(d3_start_date, "ns")

def group_by_dekad(
    ds: xr.DataArray | xr.Dataset,
) -> DataArrayGroupBy | DatasetGroupBy:
    """
    Group a dataset or array by dekad.

    Parameters
    ----------
    ds : xr.DataArray | xr.Dataset
        Dataset or DataArray to group

    Returns
    -------
    xr.core.groupby.DataArrayGroupBy | xr.core.groupby.DatasetGroupBy
        Groupby oject
    """
    group = xr.DataArray(
        data=np.vectorize(get_dekad)(ds.time.values),
        coords=ds.time.coords,
        dims=ds.time.dims,
        name="dekad",
        attrs=ds.time.attrs,
    )
    grouped_by_dekad = ds.groupby(group)
    return grouped_by_dekad

def get_dekad_no_in_month(date: np.datetime64) -> int:
    """
    Get the number of the dekad in a month that a date belongs to.
    Every month has three dekads, such that the first two dekads
    have 10 days (i.e., 1-10, 11-20), and the third is comprised of the
    remaining days of the month.

    Parameters
    ----------
    date : np.datetime64
        Date to check.

    Returns
    -------
    int
        Number of the dekad in a month.
    """
    timestamp = pd.Timestamp(date)
    year = timestamp.year
    month = timestamp.month

    first_day = datetime(year, month, 1)
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])

    d1_start_date, d2_start_date, d3_start_date = pd.date_range(
        start=first_day, end=last_day, freq="10D", inclusive="left"
    )

    if d1_start_date <= timestamp < d2_start_date:
        return 1
    elif d2_start_date <= timestamp < d3_start_date:
        return 2
    else:
        return 3


def get_dekad_no_in_year(date: np.datetime64) -> int:
    """
    Get the number of the dekad in a year that a date belongs to.
    Every month has three dekads, such that the first two dekads
    have 10 days (i.e., 1-10, 11-20), and the third is comprised of the
    remaining days of the month (21-last day). Every year has 36 dekads.

    Parameters
    ----------
    date : np.datetime64
        Date to check.

    Returns
    -------
    int
        Number of the dekad in a year.
    """
    dekad_no_in_month = get_dekad_no_in_month(date=date)
    timestamp = pd.Timestamp(date)
    month = timestamp.month
    dekad_no_in_year = ((month - 1) * 3) + dekad_no_in_month
    return dekad_no_in_year


def get_interest_period(dekad: np.datetime64, ip: int) -> list[np.datetime64]:
    """
    Get all the dekads in the interest period for a dekad.
    `dekad` will always be the end dekad of the interest period.

    Parameters
    ----------
    dekad : np.datetime64
        Dekad to get the interest period for.
        Will always be the end dekad of the interest period
    ip : int
        Number of dekads in an interest period.

    Returns
    -------
    list[np.datetime64]
        All the dekads in the interest period.
    """
    year = pd.Timestamp(dekad).year
    dekad_no_in_year = get_dekad_no_in_year(dekad) - (ip - 1)
    while dekad_no_in_year <= 0:
        year -= 1
        dekad_no_in_year += 36

    month = (dekad_no_in_year - 1) // 3 + 1
    dekad_no_in_month = (dekad_no_in_year - 1) % 3 + 1
    if dekad_no_in_month == 1:
        day = 1
    elif dekad_no_in_month == 2:
        day = 11
    elif dekad_no_in_month == 3:
        day = 21

    start_dekad = np.datetime64(datetime(year, month, day), "ns")
    date_range = pd.date_range(
        start_dekad, dekad, freq=timedelta(days=1), inclusive="both"
    ).values
    interest_period_dekads = np.unique(np.vectorize(get_dekad)(date_range))
    return interest_period_dekads


def bin_by_interest_period(
    ds: xr.Dataset | xr.DataArray, ip: int
) -> dict[np.datetime64, xr.Dataset | xr.DataArray]:
    """
    Bin each dekad in the dataset by interest period.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Dataset to bin
    ip : int
        Number of dekads in an interest period.

    Returns
    -------
    list[tuple[np.datetime64, np.datetime64]]
        List of dekad ranges to bin
    """
    start_date = ds.dekad.min().values
    end_date = ds.dekad.max().values
    date_range = pd.date_range(
        start_date, end_date, freq=timedelta(days=1), inclusive="both"
    ).values
    dekads = np.unique(np.vectorize(get_dekad)(date_range))
    bins = {i: get_interest_period(dekad=i, ip=ip) for i in dekads}
    binned_by_interest_period = {interest_period_label : ds.reindex(dekad=interest_period) for interest_period_label, interest_period in bins.items()}
    return binned_by_interest_period

def get_no_data_mask(arr):
    """
    Check if all values in an array are NaN
    """
    return np.all(np.isnan(arr))

# From https://www.geeksforgeeks.org/maximum-consecutive-ones-or-zeros-in-a-binary-array/
def max_consecutive_ones(arr: np.ndarray) -> int:
    """
    Get the maximum number of successive ones in an array.

    Parameters
    ----------

    arr : np.ndarray
        Array to check

    Returns
    -------
    int
        Maximum number of consecutive ones in the input array.

    """

    n = len(arr)
    # initialize count
    count = 0
    # initialize max
    result = 0

    for i in range(0, n):
        # If 1 is found, increment count
        # and update result if count
        # becomes more.
        if arr[i] == 1:
            # increase count
            count += 1
            result = max(result, count)
        # Reset count if one is not found
        else:
            count = 0

    return result

def read_table(path: str) -> pd.DataFrame:
    """Read a Parquet file with Conflux metadata.

    Arguments
    ---------
    path : str
        Path to Parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame with attrs set.
    """
    table = pq_read_table(path)
    df = table.to_pandas()
    meta_json = table.schema.metadata[PARQUET_META_KEY]
    metadata = json.loads(meta_json)
    for key, val in metadata.items():
        df.attrs[key] = val
    return df

def get_max_value(index, da):
    return da.isel(time_lag=index)

def get_correlation(reference_drought_index, comparison_drought_index):
    # Identify pixels that are NaN across all dekads
    # in the comparison index.
    all_nan_mask = comparison_drought_index.isnull().all(dim='dekad')

    lags=[0, 10]
    corr_list = []
    for time_lag in range(lags[0], lags[1]):
        # Modify the time lag
        time_lag += abs(lags[0]) + 1
        # Pearson's correlation coefficient
        corr = xr.corr(reference_drought_index, comparison_drought_index.shift(dekad=time_lag), dim="dekad")
        modified_corr = np.abs(corr).assign_coords(time_lag=time_lag).expand_dims({"time_lag": 1})
        corr_list.append(modified_corr)

    da_corr = xr.concat(corr_list, dim="time_lag")

    # Get the maximum modified correlation value for each pixel.
    max_corr = da_corr.max(dim="time_lag", skipna=True)

    # For each pixel get the the time lag at which the maximum
    # correlation value occurs.
    max_time_lag = xr.apply_ufunc(
            get_max_value,
            # Replace NaNs with -np.inf to ensure they are ignored in the argmax calculation.
            da_corr.fillna(-np.inf).argmax(dim="time_lag"),
            kwargs={"da": da_corr.time_lag},
            vectorize=True,
            dask="allowed",)
    max_time_lag = max_time_lag.where(~all_nan_mask)
    return dict(corr=max_corr, lag=max_time_lag)