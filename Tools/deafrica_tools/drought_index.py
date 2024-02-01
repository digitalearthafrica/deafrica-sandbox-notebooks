# Load python packages.

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from deafrica_tools.load_wapor import get_dekad_start_dates


def bin_by_dekad(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Bin a timeseries by dekad (10-day) intervals.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        Timeseries to bin

    Returns
    -------
    xr.Dataset | xr.DataArray
        Dekadal timeseries.
    """
    # Get the dekadal (10-day) rainfall record.
    years = np.sort(np.unique(ds.time.dt.year.values))
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    bin_labels = []
    for year in years:
        for month in months:
            # Each month has 3 dekads.
            bin_labels.extend(get_dekad_start_dates(year=year, month=month))

    bins = bin_labels.copy()
    bins.append(pd.Timestamp(year=year + 1, month=1, day=1))

    # Resample the dataset using the bins
    ds_resampled = ds.groupby_bins("time", bins, labels=bin_labels, right=False).mean()
    ds_resampled = ds_resampled.rename({"time_bins": "time"})

    # Drop values only if all values along the time dimension are NaN:
    ds_resampled = ds_resampled.dropna(dim="time", how="all")
    return ds_resampled


def get_dekad_no_in_month(date: str | datetime | pd.Timestamp) -> int:
    """
    Get the dekad number in a month for a date. A month has 3 dekads.

    Parameters
    ----------
    date : str | datetime | pd.Timestamp
        Date to parse

    Returns
    -------
    int
        Dekad number in the month.
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
        return 1
    elif d2_start_date <= timestamp < d3_start_date:
        return 2
    else:
        return 3


def get_dekad_no_in_year(date: str | datetime | pd.Timestamp) -> int:
    """
    Get the dekad number in a year for a date. A year has 36 dekads.

    Parameters
    ----------
    date : str | datetime | pd.Timestamp
        Date to parse

    Returns
    -------
    int
        Dekad number in the year.
    """
    if isinstance(date, pd.Timestamp):
        timestamp = date
    else:
        timestamp = pd.Timestamp(date)

    dekad_no_in_month = get_dekad_no_in_month(date=timestamp)

    dekad_no_in_year = ((timestamp.month - 1) * 3) + dekad_no_in_month

    return dekad_no_in_year


def group_by_dekad_no_in_year(ds) -> dict:
    """
    Group a timeseries by dekad in the year.

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Timeseries to group.

    Returns
    -------
    dict
        Mapping from group labels to indices.

    """
    # Vectorize the get_dekad_no_in_year function
    vfunc = np.vectorize(get_dekad_no_in_year)
    # Apply the vectorize function to the time values of the timeseries
    # and create an Array whose unique values should be used to group the timeseries
    group = xr.DataArray(
        data=vfunc(ds.time.values),
        coords=ds.time.coords,
        dims=ds.time.dims,
        name="dekad_no_in_year",
        attrs=ds.time.attrs,
    )

    groups = ds.groupby(group).groups

    return groups


def get_data_for_ip(
    ds: xr.Dataset | xr.DataArray,
    y: int,
    d: int,
    ip: int,
):
    """
    Get the timeseries data covering the interest period.

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Dekadal (10-day) timeseries
    y: int
        Year of interest
    d: int
        Dekad of interest in the year of interest.
        Note: A year has 36 dekads following a 1-based index i.e. 1 to 36
    ip: int
        Interest period e.g. 3,4,5 dekads

    Returns
    -------
    xr.Dataset | xr.DataArray
        Average for the interest period.
    """
    ds_list = []
    for j in range(0, (ip - 1) + 1):
        if d - j < 1:
            # Get data from the previous year.
            year = y - 1
            dekad = d - j + 36
        else:
            year = y
            dekad = d - j

        try:
            sel_by_year = ds.sel(time=str(year))
        except KeyError:
            print(f"No data available for year {year}")
            continue
        else:
            try:
                sel_by_dekad = sel_by_year.isel(time=group_by_dekad_no_in_year(sel_by_year)[dekad])
            except KeyError:
                print(f"No data available for year {year} dekad {dekad}")
                continue
            else:
                ds_list.append(sel_by_dekad)

    if ds_list:
        ds_merged = xr.concat(ds_list, dim="time").sortby("time")
    else:
        if isinstance(ds, xr.Dataset):
            ds_merged = xr.Dataset()
        elif isinstance(ds, xr.DataArray):
            ds_merged = xr.DataArray()

    return ds_merged


def get_actual_avg_for_ip(
    ds: xr.Dataset | xr.DataArray,
    y: int,
    d: int,
    ip: int,
):
    """
    Get the actual average for the interest period.

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Dekadal (10-day) timeseries
    y: int
        Year of interest
    d: int
        Dekad of interest in the year of interest.
        Note: A year has 36 dekads following a 1-based index i.e. 1 to 36
    ip: int
        Interest period e.g. 3,4,5 dekads

    Returns
    -------
    xr.Dataset | xr.DataArray
        Average for the interest period.
    """
    # Get the data for the interest period.
    ds_ip = get_data_for_ip(ds=ds, y=y, d=d, ip=ip)

    if ds_ip:
        # Get the average over the interest period
        actual_avg_for_ip = ds_ip.sum(dim="time") / ip
    else:
        if isinstance(ds_ip, xr.Dataset):
            actual_avg_for_ip = xr.Dataset()
        elif isinstance(ds_ip, xr.DataArray):
            actual_avg_for_ip = xr.DataArray()
    return actual_avg_for_ip


def get_longterm_avg_for_ip(
    ds: xr.Dataset | xr.DataArray,
    d: int,
    ip: int,
):
    """
    Get the long term average for the interest period.

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Dekadal (10-day) timeseries
    d: int
        Dekad of interest in the year of interest.
        Note: A year has 36 dekads following a 1-based index i.e. 1 to 36
    ip: int
        Interest period e.g. 3,4,5 dekads

    Returns
    -------
    xr.Dataset | xr.DataArray
        Long term average for the interest period over the years of available data.
    """
    # Number of years with data
    years = np.sort(np.unique(ds.time.dt.year.values))
    n = len(years)

    sum_k = 0
    for y in years:
        # Calculate the actual average for the interest period for each year.
        actual_avg_for_ip_for_year = get_actual_avg_for_ip(ds=ds, y=y, d=d, ip=ip)
        if actual_avg_for_ip_for_year:
            sum_k = sum_k + actual_avg_for_ip_for_year

    longterm_avg_for_ip = sum_k / n

    return longterm_avg_for_ip


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
        # Reset count when 0 is found
        if arr[i] == 0:
            count = 0

        # If 1 is found, increment count
        # and update result if count
        # becomes more.
        else:
            # increase count
            count += 1
            result = max(result, count)

    return result


def get_actual_run_length_in_ip(
    ds: xr.Dataset | xr.DataArray,
    y: int,
    d: int,
    ip: int,
    inverse: bool = False,
):
    """
    Get the actual run length in the interest period.
    Run length is the maximum number of sucessive dekads below (inverse==False)
    or above (inverse==True) the long term average in the interest period.
    (length of continuous deficit  or excess in the interest period)

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Dekadal (10-day) timeseries
    y: int
        Year of interest
    d: int
        Dekad of interest in the year of interest.
        Note: A year has 36 dekads following a 1-based index i.e. 1 to 36
    ip: int
        Interest period e.g. 3,4,5 dekads
    inverse: bool
        Whether to calculate continuous deficit or excess, by default False.
        If inverse==False run length is the maximum number of sucessive dekads below
        the long term average in the interest period.
        If inverse==True run length is the maximum number of sucessive dekads above
        the long term average in the interest period.
    Returns
    -------
    xr.Dataset | xr.DataArray
        Run length in the interest period.
    """
    # Get the data for the interest period from the timeseries.
    ds_ip = get_data_for_ip(ds=ds, y=y, d=d, ip=ip)

    if ds_ip:
        # Get the long term average for the interest period.
        longterm_avg_for_ip = get_longterm_avg_for_ip(ds=ds, d=d, ip=ip)

        if inverse:
            ds_ip_masked = xr.where(ds_ip > longterm_avg_for_ip, 1, 0)
        else:
            ds_ip_masked = xr.where(ds_ip < longterm_avg_for_ip, 1, 0)

        # Can also use numpy.apply_along_axis
        actual_run_length_in_ip = xr.apply_ufunc(
            max_consecutive_ones, ds_ip_masked, input_core_dims=[["time"]], vectorize=True
        )

        # Modify the run  length.
        mod_actual_run_length_in_ip =  actual_run_length_in_ip

    else:
        if isinstance(ds_ip, xr.Dataset):
            actual_run_length_in_ip = xr.Dataset()
        elif isinstance(ds_ip, xr.DataArray):
            actual_run_length_in_ip = xr.DataArray()
    return actual_run_length_in_ip


def get_longterm_avg_for_run_length_in_ip(
    ds: xr.Dataset | xr.DataArray,
    d: int,
    ip: int,
    inverse: bool = False,
):
    """
    Get the long term average for the run length in the interest period.
    Run length is the maximum number of sucessive dekads below (inverse==False)
    or above (inverse==True) the long term average in the interest period.
    (length of continuous deficit  or excess in the interest period)

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Dekadal (10-day) timeseries
    d: int
        Dekad of interest in the year of interest.
        Note: A year has 36 dekads following a 1-based index i.e. 1 to 36
    ip: int
        Interest period e.g. 3,4,5 dekads
    inverse: bool
        Whether to calculate continuous deficit or excess, by default False.
        If inverse==False run length is the maximum number of sucessive dekads below
        the long term average in the interest period.
        If inverse==True run length is the maximum number of sucessive dekads above
        the long term average in the interest period.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Long term average for run length in the interest period over the years of available data.
    """
    # Number of years with data
    years = np.sort(np.unique(ds.time.dt.year.values))
    n = len(years)

    # Note: not using the get_actual_run_length_in_ip function for each year
    # because of the repeated calculation of the long term mean
    longterm_avg_for_ip = get_longterm_avg_for_ip(ds=ds, d=d, ip=ip)

    sum_k = 0
    for y in years:
        # Get the data for the interest period from the timeseries.
        ds_ip_y = get_data_for_ip(ds=ds, y=y, d=d, ip=ip)

        if ds_ip_y:
            if inverse:
                ds_ip_masked_y = xr.where(ds_ip_y > longterm_avg_for_ip, 1, 0)
            else:
                ds_ip_masked_y = xr.where(ds_ip_y < longterm_avg_for_ip, 1, 0)

            # Can also use numpy.apply_along_axis
            actual_run_length_in_ip_y = xr.apply_ufunc(
                max_consecutive_ones, ds_ip_masked_y, input_core_dims=[["time"]], vectorize=True
            )
            if actual_run_length_in_ip_y:
                sum_k = sum_k + actual_run_length_in_ip_y

    longterm_avg_for_run_length_in_ip = sum_k / n

    return longterm_avg_for_run_length_in_ip
