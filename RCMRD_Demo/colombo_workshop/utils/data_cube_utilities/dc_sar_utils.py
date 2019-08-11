import xarray as xr
import numpy as np


#db is given by 10*log10(DN) + CF. for ALOS this is apparently -83, 0 for s1
#modifies in place.
def dn_to_db(dataset_in, data_vars=['hh', 'hv'], cf=-83):
    for data_var in data_vars:
        dataset_in[data_var] = (
            10 * xr.ufuncs.log10(xr.ufuncs.square(dataset_in[data_var].astype('float64'))) + cf).astype('float32')
