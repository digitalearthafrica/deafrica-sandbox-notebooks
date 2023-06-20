import xarray as xr
import numpy as np

# function to load soil moisture data

def load_soil_moisture(lat, lon, time, product = 'surface', grid = 'nearest'):
    product_baseurl = 'https://dapds00.nci.org.au/thredds/dodsC/ub8/global/GRAFS/'
    assert product in ['surface', 'rootzone'], 'product parameter must be surface or root-zone'
    # lat, lon grid
    if grid == 'nearest':
        # select lat/lon range from data; snap to nearest grid
        lat_range, lon_range = None, None
    else:
        # define a grid that covers the entire area of interest
        lat_range = np.arange(np.max(np.ceil(np.array(lat)*10.+0.5)/10.-0.05), np.min(np.floor(np.array(lat)*10.-0.5)/10.+0.05)-0.05, -0.1)
        lon_range = np.arange(np.min(np.floor(np.array(lon)*10.-0.5)/10.+0.05), np.max(np.ceil(np.array(lon)*10.+0.5)/10.-0.05)+0.05, 0.1)
    # split time window into years
    day_range = np.array(time).astype("M8[D]")
    year_range = np.array(time).astype("M8[Y]")
    if product == 'surface':
        product_name = 'GRAFS_TopSoilRelativeWetness_'
    else: product_name = 'GRAFS_RootzoneSoilWaterIndex_'
    datasets = []
    for year in np.arange(year_range[0], year_range[1]+1, np.timedelta64(1, 'Y')):
        start = np.max([day_range[0], year.astype("M8[D]")])
        end = np.min([day_range[1], (year+1).astype("M8[D]")-1])
        product_url = product_baseurl + product_name +'%s.nc'%str(year)
        print(product_url)
        # data is loaded lazily through OPeNDAP
        ds = xr.open_dataset(product_url)
        if lat_range is None:
            # select lat/lon range from data if not specified; snap to nearest grid
            test = ds.sel(lat=list(lat), lon=list(lon), method='nearest')
            lat_range = slice(test.lat.values[0], test.lat.values[1])
            lon_range = slice(test.lon.values[0], test.lon.values[1])
        # slice before return
        ds = ds.sel(lat=lat_range, lon=lon_range, time=slice(start, end)).compute()
        datasets.append(ds)
    return xr.merge(datasets)