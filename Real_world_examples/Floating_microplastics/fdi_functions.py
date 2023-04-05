import xarray as xr

# Function to normalize the band values. 
def normalize_bands(ds):
    bands = list(ds.data_vars)
    for band in bands:
        ds[band] = (ds[band] - ds[band].min()) / (ds[band].max() - ds[band].min())
    return ds

def normalize_intensity(ds):
    if "time" in ds.dims:
        time_steps = ds.time.values

        ds_list = []
        for time in time_steps:
            ds_time = ds.sel(time=time)
            ds_time = normalize_bands(ds_time)
            ds_list.append(ds_time)

        ds = xr.concat(objs=ds_list, dim="time")
        
    else:
        ds = normalize_bands(ds)

    return ds


# Function to calculate the FAI, FDI and PI indices.
def insert_indices(ds):
    wavelength_nir = 842 * 1e-9
    wavelength_red = 665 * 1e-9
    wavelength_swir1 = 1610 * 1e-9
    
    # Floating Algae Index
    ds["FAI"] = (ds.nir - ds.red) + (ds.red - ds.swir_1) * (( wavelength_nir - wavelength_red) / (wavelength_swir1 - wavelength_red))
    
    # Floating Debris Index
    ds["FDI"] = ds.nir - (ds.red_edge_2 + (ds.swir_1 - ds.red_edge_2) * ((wavelength_nir - wavelength_red) / (wavelength_swir1 - wavelength_red)) * 10)
    
    # Plastic Index (PI)
    ds['PI'] = ds['nir'] / (ds['nir'] + ds['red'])
    
    return ds
