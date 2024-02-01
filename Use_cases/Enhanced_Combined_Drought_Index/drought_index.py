# Load python packages.

import calendar
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


def get_dekad_date(date: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """
    Checks the dekad of a date and returns the dekad date.

    Parameters
    ----------
    date : str | datetime | pd.Timestamp
        Date to check.

    Returns
    -------
    pd.Timestamp
        Date of the dekad.
    """
    # Get the year and month from the date.
    if isinstance(date, pd.Timestamp):
        timestamp = date
    else:
        timestamp = pd.Timestamp(date)

    year = timestamp.year
    month = timestamp.month

    # First day of the month
    start_date = datetime(year, month, 1)
    # Last day of the month.
    end_date = datetime(year, month, calendar.monthrange(year, month)[1])

    d1_start_date, d2_start_date, d3_start_date = pd.date_range(
        start=start_date, end=end_date, freq="10D", inclusive="left"
    )

    if d1_start_date <= timestamp < d2_start_date:
        return d1_start_date
    elif d2_start_date <= timestamp < d3_start_date:
        return d2_start_date
    else:
        return d3_start_date


def resample_ds(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Resample a timeseries by dekad.

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Timeseries to resample.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Resampled timeseries.

    """
    # Apply the vectorized get_dekad_date function to the time values
    # of the timeseries and create an array whose unique values should be used
    # to group the timeseries.
    group = xr.DataArray(
        data=np.vectorize(get_dekad_date)(ds.time.values),
        coords=ds.time.coords,
        dims=ds.time.dims,
        name="dekad_date",
        attrs=ds.time.attrs,
    )

    # Resample the timeseries into 10-day intervals.
    ds_resampled = ds.groupby(group).mean(dim="time").compute()
    ds_resampled = ds_resampled.rename({"dekad_date": "time"})

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

    # First day of the month
    start_date = datetime(year, month, 1)
    # Last day of the month.
    end_date = datetime(year, month, calendar.monthrange(year, month)[1])

    d1_start_date, d2_start_date, d3_start_date = pd.date_range(
        start=start_date, end=end_date, freq="10D", inclusive="left"
    )

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
    Group a timeseries by dekad number in the year.

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Timeseries to group.

    Returns
    -------
    dict
        Mapping from group labels to indices.

    """
    # Apply the vectorized get_dekad_no_in_year function to the time values of
    # the timeseries and create an Array whose unique values should be used to
    # group the timeseries
    group = xr.DataArray(
        data=np.vectorize(get_dekad_no_in_year)(ds.time.values),
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
                sel_by_dekad = sel_by_year.isel(
                    time=group_by_dekad_no_in_year(sel_by_year)[dekad]
                )
            except KeyError:
                print(f"No data available for year {year} dekad {dekad}")
                continue
            else:
                ds_list.append(sel_by_dekad)

    if ds_list:
        ds_merged = xr.concat(ds_list, dim="time").sortby("time")
        return ds_merged
    else:
        return None


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

    if ds_ip is not None:
        # Get the average over the interest period
        actual_avg_for_ip = ds_ip.mean(dim="time")
        return actual_avg_for_ip
    else:
        return None


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
    years = np.unique(ds.time.dt.year.values)

    ds_list = []
    for y in years:
        # Calculate the actual average for the interest period for each year.
        actual_avg_for_ip_for_year = get_actual_avg_for_ip(ds=ds, y=y, d=d, ip=ip)
        if actual_avg_for_ip_for_year is not None:
            ds_list.append(
                actual_avg_for_ip_for_year.assign_coords(year=y).expand_dims(
                    {"year": 1}
                )
            )

    if ds_list:
        ds_merged = xr.concat(ds_list, dim="year").sortby("year")
        longterm_avg_for_ip = ds_merged.mean("year")
        return longterm_avg_for_ip
    else:
        return None


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


def get_no_data_mask(arr):
    """
    Check if all values in an array are NaN
    """
    return np.all(np.isnan(arr))


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

    if ds_ip is not None:
        no_data_mask = xr.apply_ufunc(
            get_no_data_mask,
            ds_ip,
            input_core_dims=[["time"]],
            vectorize=True,
            dask="allowed",
        )

        # Get the long term average for the interest period.
        longterm_avg_for_ip = get_longterm_avg_for_ip(ds=ds, d=d, ip=ip)

        if inverse:
            ds_ip_masked = xr.where(ds_ip > longterm_avg_for_ip, 1, 0)
        else:
            ds_ip_masked = xr.where(ds_ip < longterm_avg_for_ip, 1, 0)

        # Can also use numpy.apply_along_axis
        actual_run_length_in_ip = xr.apply_ufunc(
            max_consecutive_ones,
            ds_ip_masked,
            input_core_dims=[["time"]],
            vectorize=True,
            dask="allowed",
        )

        mod_actual_run_length_in_ip = (
            actual_run_length_in_ip.max() + 1
        ) - actual_run_length_in_ip

        mod_actual_run_length_in_ip = mod_actual_run_length_in_ip.where(~no_data_mask)

        return mod_actual_run_length_in_ip

    else:
        return None


def get_longterm_avg_run_length_in_ip(
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
        If inverse==False run length is the maximum number of sucessive dekads
        below the long term average in the interest period.
        If inverse==True run length is the maximum number of sucessive dekads
        above the long term average in the interest period.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Long term average for run length in the interest period over the years
        of available data.
    """
    years = np.unique(ds.time.dt.year.values)

    # Note: not using the get_actual_run_length_in_ip function for each year
    # to avoid of the repeated calculation of the long term mean
    longterm_avg_for_ip = get_longterm_avg_for_ip(ds=ds, d=d, ip=ip)

    ds_list = []
    for y in years:
        # Get the data for the interest period from the timeseries.
        ds_ip_y = get_data_for_ip(ds=ds, y=y, d=d, ip=ip)

        if ds_ip_y is not None:
            no_data_mask_y = xr.apply_ufunc(
                get_no_data_mask,
                ds_ip_y,
                input_core_dims=[["time"]],
                vectorize=True,
                dask="allowed",
            )

            if inverse:
                ds_ip_masked_y = xr.where(ds_ip_y > longterm_avg_for_ip, 1, 0)
            else:
                ds_ip_masked_y = xr.where(ds_ip_y < longterm_avg_for_ip, 1, 0)

            # Can also use numpy.apply_along_axis
            actual_run_length_in_ip_y = xr.apply_ufunc(
                max_consecutive_ones,
                ds_ip_masked_y,
                input_core_dims=[["time"]],
                vectorize=True,
                dask="allowed",
            )

            mod_actual_run_length_in_ip_y = (
                actual_run_length_in_ip_y.max() + 1
            ) - actual_run_length_in_ip_y

            mod_actual_run_length_in_ip_y = mod_actual_run_length_in_ip_y.where(
                ~no_data_mask_y
            )

            ds_list.append(
                mod_actual_run_length_in_ip_y.assign_coords(year=y).expand_dims(
                    {"year": 1}
                )
            )

    if ds_list:
        ds_merged = xr.concat(ds_list, dim="year").sortby("year")
        longterm_avg_run_length_in_ip = ds_merged.mean("year")
        return longterm_avg_run_length_in_ip
    else:
        return None


def calculate_drought_index(
    ds: xr.Dataset | xr.DataArray,
    y: int,
    d: int,
    ip: int,
    inverse: bool = False,
):
    """
    Calculate the Drought Index for the year {y} and dekad {d} using an
    interest period of {ip} dekads.

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
        If inverse==False run length is the maximum number of sucessive dekads
        below the long term average in the interest period.
        If inverse==True run length is the maximum number of sucessive dekads
        above the long term average in the interest period.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Drough Index
    """

    actual_avg_for_ip = get_actual_avg_for_ip(ds=ds, y=y, d=d, ip=ip)

    longterm_avg_for_ip = get_longterm_avg_for_ip(ds=ds, d=d, ip=ip)

    actual_run_length_in_ip = get_actual_run_length_in_ip(ds=ds, y=y, d=d, ip=ip)

    longterm_avg_run_length_in_ip = get_longterm_avg_run_length_in_ip(
        ds=ds, d=d, ip=ip, inverse=inverse
    )

    DI = (actual_avg_for_ip / longterm_avg_for_ip) * np.sqrt(
        actual_run_length_in_ip / longterm_avg_run_length_in_ip
    )

    DI_scaled = (DI - DI.min()) / (DI.max() - DI.min())

    return DI_scaled
