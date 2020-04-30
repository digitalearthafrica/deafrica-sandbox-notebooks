import numpy as np

def TIMESAT_stats(dataarray, time_dim='time'):
    """
    For a 1D array of values for a vegetation index - for which higher values tend to 
    indicate more vegetation - determine several statistics:
    1. Beginning of Season (BOS): The time index of the beginning of the growing season.
        (The downward inflection point before the maximum vegetation index value)
    2. End of Season (EOS): The time index of the end of the growing season.
        (The upward inflection point after the maximum vegetation index value)
    3. Middle of Season (MOS): The time index of the maximum vegetation index value.
    4. Length of Season (EOS-BOS): The time length of the season (index difference).
    5. Base Value (BASE): The minimum vegetation index value.
    6. Max Value (MAX): The maximum vegetation index value (the value at MOS).
    7. Amplitude (AMP): The difference between BASE and MAX.
    
    Parameters
    ----------
    dataarray: xarray.DataArray
        The 1D array of non-NaN values to determine the statistics for.
    time_dim: string
        The name of the time dimension in `dataarray`.

    Returns
    -------
    stats: dict
        A dictionary mapping statistic names to values.
    """
    assert time_dim in dataarray.dims, "The parameter `time_dim` is \"{}\", " \
        "but that dimension does not exist in the data.".format(time_dim)
    stats = {}
    data_np_arr = dataarray.values
    time_np_arr = dataarray[time_dim].values
    data_inds = np.arange(len(data_np_arr))
    
    # Obtain the first and second derivatives.
    fst_deriv = np.gradient(data_np_arr, time_np_arr)
    pos_fst_deriv = fst_deriv > 0
    neg_fst_deriv = 0 > fst_deriv
    snd_deriv = np.gradient(fst_deriv, time_np_arr)
    pos_snd_deriv = snd_deriv > 0
    neg_snd_deriv = 0 > snd_deriv
    
    # Determine MOS.
    # MOS is the index of the highest value immediately preceding a transition
    # of the first derivative from positive to negative.
    pos_to_neg_fst_deriv = pos_fst_deriv.copy()
    for i in range(len(pos_fst_deriv)):
        if i == len(pos_fst_deriv) - 1: # last index
            pos_to_neg_fst_deriv[i] = False
        elif pos_fst_deriv[i] and not pos_fst_deriv[i+1]: # + to -
            pos_to_neg_fst_deriv[i] = True
        else: # everything else
            pos_to_neg_fst_deriv[i] = False
    idxmos_potential_inds = data_inds[pos_to_neg_fst_deriv]
    idxmos_subset_ind = np.nanargmax(data_np_arr[pos_to_neg_fst_deriv])
    idxmos = idxmos_potential_inds[idxmos_subset_ind]
    stats['Middle of Season'] = idxmos
    
    data_inds_after_mos = np.roll(data_inds, len(data_inds)-idxmos-1)
    
    # Determine BOS.
    # BOS is the first negative inflection point of the positive values 
    # of the first derivative starting after and ending at the MOS.
    idxbos = data_inds_after_mos[np.nanargmax((pos_fst_deriv & neg_snd_deriv)[data_inds_after_mos])]
    stats['Beginning of Season'] = idxbos
    
    # Determine EOS.
    # EOS is the last positive inflection point of the negative values 
    # of the first derivative starting after and ending at the MOS.
    idxeos = data_inds_after_mos[np.nanargmax((neg_fst_deriv & pos_snd_deriv)[data_inds_after_mos][::-1])]
    stats['End of Season'] = idxeos
    
    # Determine EOS-BOS.
    stats['Length of Season'] = idxeos - idxbos
    # Determine BASE.
    stats['Base Value'] = data_np_arr.min()
    # Determine MAX.
    stats['Max Value'] = data_np_arr.max()
    # Determine AMP.
    stats['Amplitude'] = stats['Max Value'] - stats['Base Value']
    
    return stats