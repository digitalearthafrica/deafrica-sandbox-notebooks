# deafrica_phenology.py
"""
This script contains functions for calculating land-surface 
phenology metrics on a time series of a vegetations index
stored within an xarray.DataArray.  
"""


import sys
import numpy as np
import xarray as xr
from scipy.stats import skew
sys.path.append('../Scripts')
from deafrica_datahandling import first, last


def allNaN_arg(xarr, dim, stat):
    """
    Calculate da.argmax() or da.argmin() while handling
    all-NaN slices. Fills all-NaN locations with an
    integer and then masks the offending cells.
    
    Value of the fill_na() will never be returned as index 
    of argmax/min as fill value exceeds the min/max
    value of the array.
    
    Params
    ------
    xarr : xarray.DataArray
    dim : str, 
            Dimension over which to calculate argmax, argmin e.g. 'time'
    stat : str,
        The statistic to calculte, either 'min' for argmin()
        or 'max' for .argmax()
    
    Returns
    ------
    xarray.DataArray
    
    """
    #generate a mask where entire axis along dimension is NaN
    mask = xarr.min(dim=dim, skipna=True).isnull()

    if stat == 'max':
        y = xarr.fillna(float(xarr.min() - 1))
        y = y.argmax(dim=dim, skipna=True).where(~mask)
        return y

    if stat == 'min':
        y = xarr.fillna(float(xarr.max() + 1))
        y = y.argmin(dim=dim, skipna=True).where(~mask)
        return y

def _vpos(da):
    """
    vPOS = Value at peak of season
    """
    return da.max('time')

def _pos(da):
    """
    POS = DOY of peak of season
    """
    return da.isel(time=da.argmax('time')).time.dt.dayofyear

def _trough(da):
    """
    Trough = Minimum value
    """
    return da.min('time')

def _aos(vpos, trough):
    """
    AOS = Amplitude of season
    """
    return vpos - trough

def _vsos(da, pos, method_sos='first'):
    """
    vSOS = Value at the start of season
    
    Params
    -----
    da : xarray.DataArray
    method_sos : str, 
        If 'first' then vSOS is estimated
        as the first positive slope on the
        greening side of the curve. If 'median',
        then vSOS is estimated as the median value
        of the postive slopes on the greening side
        of the curve.
        
    """
    # select timesteps before peak of season (AKA greening)
    greenup = da.where(da.time < pos.time)
    # find the first order slopes
    green_deriv = greenup.differentiate('time')
    # find where the first order slope is postive
    pos_green_deriv = green_deriv.where(green_deriv > 0)

    if method_sos == 'first':
        # get the timestep where slope first becomes positive
        # to estimate the DOY when growing season starts
        #using the compositing method 'first'
        return first(pos_green_deriv, dim='time')

    if method_sos == 'median':
        #positive slopes on greening side
        pos_greenup = greenup.where(pos_green_deriv)
        #find the median
        median = pos_greenup.median('time')
        #distance of values from median
        distance = pos_greenup - median
        #find index (argmin) where distance is smallest (ie this
        #is where the median is for each pixel)
        idx = allNaN_arg(distance, 'time', 'min').astype('int16')
        return pos_greenup.isel(time=idx)

def _sos(vsos):
    """
    SOS = DOY for start of season
    """
    return vsos.time.dt.dayofyear

def _veos(da, pos, method_eos='last'):
    """
    vEOS = Value at the start of season
    
    Params
    -----
    method_eos : str
        If 'first' then vEOS is estimated
        as the last negative slope on the
        senescing side of the curve. If 'median',
        then vEOS is estimated as the 'median' value
        of the negative slopes on the senescing 
        side of the curve.
    """
    # select timesteps before peak of season (AKA greening)
    senesce = da.where(da.time > pos.time)
    # find the first order slopes
    senesce_deriv = senesce.differentiate('time')
    # find where the fst order slope is postive
    neg_senesce_deriv = senesce_deriv.where(senesce_deriv < 0)

    if method_eos == 'last':
        # get the timestep where slope is last negative to estimate
        # the DOY when growing season ends
        return last(neg_senesce_deriv, dim='time')

    if method_eos == 'median':
        #negative slopes on senescing side
        neg_senesce = senesce.where(neg_senesce_deriv)
        #find medians
        median = neg_senesce.median('time')
        #distance to the median
        distance = neg_senesce - median
        #index where median occurs
        idx = allNaN_arg(distance, 'time', 'min').astype('int16')
        return neg_senesce.isel(time=idx)

def _eos(veos):
    """
    EOS = DOY for end of seasonn
    """
    return veos.time.dt.dayofyear


def _los(da, eos, sos):
    """
    LOS = Length of season (in DOY)
    """
    los = eos - sos
    #handle negative values
    los = xr.where(los >= 0,
                   los, 
                   da.time.dt.dayofyear.values[-1] + (eos.where(los < 0) - sos.where(los < 0)))
    
    return los


def _rog(vpos, vsos, pos, sos):
    """
    ROG = Rate of Greening (Days)
    """
    return (vpos - vsos) / (pos - sos)


def _ros(veos, vpos, eos, pos):
    """
    ROG = Rate of Senescing (Days)
    """
    return (veos - vpos) / (eos - pos)


def xr_phenology(da,
                 stats=[
                     'SOS', 'POS', 'EOS', 'Trough',
                     'vSOS', 'vPOS', 'vEOS', 'LOS',
                     'AOS', 'ROG', 'ROS'
                 ],
                 method_sos='median',
                 method_eos='median',
                 interpolate=False,
                 interpolate_na=False,
                 interp_method="linear",
                 interp_interval='2W'):
    
    """
    Obtain land surface phenology metrics from an
    xarray.DataArray containing a timeseries of a 
    vegetation index like NDVI.
    
    last modified May 2020
    
    Parameters
    ----------
    da :  xarray.DataArray
        DataArray should contain a 2 or 3D time series of a
        vegetation index like NDVI
    stats : list
        list of phenological statistics to return. Regardless of
        the metrics returned, all statistics are calculated
        due to inter-dependencies between metrics.
        Options include:
            SOS = DOY of start of season
            POS = DOY of peak of season
            EOS = DOY of end of season
            vSOS = Value at start of season
            vPOS = Value at peak of season
            vEOS = Value at end of season
            Trough = Minimum value of season
            LOS = Length of season (DOY)
            AOS = Amplitude of season (in value units)
            ROG = Rate of greening
            ROS = Rate of senescence
    method_sos : str 
        If 'first' then vSOS is estimated
        as the first positive slope on the
        greening side of the curve. If 'median',
        then vSOS is estimated as the median value
        of the postive slopes on the greening side
        of the curve.
    method_eos : str
        If 'first' then vEOS is estimated
        as the last negative slope on the
        senescing side of the curve. If 'median',
        then vEOS is estimated as the 'median' value
        of the negative slopes on the senescing 
        side of the curve.
    interpolate : bool
        Whether to interpolate the time dimension of the 
        dataset using xarray's inbuilt .resample().interpolate()
        methods. This can be helpful if the timeseries is sparse.
    interpolate_na : bool
        Whether to fill NaN values in the dataset using an interpolation
        method. Can be used in conjunction with interpolate=True,
        in which case the NaNs are filled before the time series
        is interpolated. Note this can be very slow.
    interp_method : str
        Which method to use for the `interpolate` and `interpolate_na`
        parameters. Options include 'linear' or 'nearest'.
    interp_interval : str
        The time interval to interpolate too. e.g '1D', '1W', 1M, '1Y'
    
    Outputs
    -------
        xarray.Dataset containing variables for the selected 
        phenology statistics 
        
    """
    #Check parameters before running calculations
    if interp_method not in ('linear', 'nearest'):
         raise ValueError("Currently only interp_methods 'nearest' and 'linear' are supported")
            
    if method_sos != 'median':
        raise ValueError("Currently only method_sos 'median' is supported")
    
    if method_eos != 'median':
        raise ValueError("Currently only method_eos 'median' is supported")
            
    # If stats supplied is not a list, convert to list.
    stats = stats if isinstance(stats, list) else [stats]
    
    #Interpolate and/or fill NaNs
    if (interpolate_na == True) & (interpolate == True):
        print('removing NaNs')
        da = da.interpolate_na(dim='time', method=interp_method)

        #resample time dim and interpolate values
        da = da.resample(time=interp_interval).interpolate(interp_method)
        print("    Interpolated dataset to " + str(len(da.time)) +
              " time-steps")

    if (interpolate_na == False) & (interpolate == True):
        da = da.resample(time=interp_interval).interpolate(interp_method)
        print("Interpolated dataset to " + str(len(da.time)) + " time-steps")

    if (interpolate_na == True) & (interpolate == False):
        print('removing NaNs')
        da = da.interpolate_na(dim='time', method=interp_method)

    vpos = _vpos(da)
    pos = _pos(da)
    trough = _trough(da)
    aos = _aos(vpos, trough)
    vsos = _vsos(da, pos, method_sos=method_sos)
    sos = _sos(vsos)
    veos = _veos(da, pos, method_eos=method_eos)
    eos = _eos(veos)
    los = _los(da, eos, sos)
    rog = _rog(vpos, vsos, pos, sos)
    ros = _ros(veos, vpos, eos, pos)

    # Dictionary containing the statistics
    stats_dict = {
        'SOS': sos,
        'EOS': eos,
        'vSOS': vsos,
        'vPOS': vpos,
        'Trough': trough,
        'POS': pos,
        'vEOS': veos,
        'LOS': los,
        'AOS': aos,
        'ROG': rog,
        'ROS': ros,
    }

    #intialise dataset with first statistic
    ds = stats_dict[stats[0]].to_dataset(name=stats[0])

    #add the other stats to the dataset
    for stat in stats[1:]:
        stats_keep = stats_dict.get(stat)
        ds[stat] = stats_dict[stat]

    return ds
