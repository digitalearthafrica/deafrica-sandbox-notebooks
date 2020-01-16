import time
import numpy as np
import xarray as xr

from dc_utilities import _get_transform_from_xr
import dc_utilities
import datacube
import rasterio

## Export ##

def export_xarray_to_netcdf(data, path):
    """
    Exports an xarray.Dataset as a single NetCDF file.
    All attributes except CRS will be lost, and the CRS
    attribute will be converted to a string.

    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray
        The Dataset or DataArray to export.
    path: str
        The path to store the exported NetCDF file at.
        Must include the filename and ".nc" extension.
    """
    # To be able to call `xarray.Dataset.to_netcdf()`, convert the CRS
    # object from the Data Cube to a string and remove all other attributes.
    for attr in data.attrs:
        if attr == 'crs' and not isinstance(attr, str):
            data.attrs['crs'] = data.crs.crs_str
        else:
            del data.attrs[attr]
    if isinstance(data, xr.Dataset):
        for data_var in data.data_vars:
            for attr in list(data[data_var].attrs):
                if attr == 'crs' and not isinstance(attr, str):
                    data[data_var].attrs['crs'] = data[data_var].crs.crs_str
                else:
                    del data[data_var].attrs[attr]
    if 'time' in data.coords:
        if 'units' in data.time.attrs:
            time_units = data.time.attrs['units']
            del data.time.attrs['units']
            data.time.encoding['units'] = time_units
    data.to_netcdf(path)

def export_slice_to_geotiff(ds, path, x_coord='longitude', y_coord='latitude'):
    """
    Exports a single slice of an xarray.Dataset as a GeoTIFF.

    ds: xarray.Dataset
        The Dataset to export. Must have exactly 2 dimensions - 'latitude' and 'longitude'.
    x_coord, y_coord: string
        Names of the x and y coordinates in `ds`.
    path: str
        The path to store the exported GeoTIFF.
    """
    kwargs = dict(tif_path=path, data=ds.astype(np.float32), bands=list(ds.data_vars.keys()),
                  x_coord=x_coord, y_coord=y_coord)
    if 'crs' in ds.attrs:
        kwargs['crs'] = str(ds.attrs['crs'])
    dc_utilities.write_geotiff_from_xr(**kwargs)


def export_xarray_to_multiple_geotiffs(ds, path, x_coord='longitude', y_coord='latitude'):
    """
    Exports an xarray.Dataset as individual time slices - one GeoTIFF per time slice.

    Parameters
    ----------
    ds: xarray.Dataset
        The Dataset to export. Must have exactly 3 dimensions - 'latitude', 'longitude', and 'time'.
        The 'time' dimension must have type `numpy.datetime64`.
    path: str
        The path prefix to store the exported GeoTIFFs. For example, 'geotiffs/mydata' would result in files named like
        'mydata_2016_12_05_12_31_36.tif' within the 'geotiffs' folder.
    x_coord, y_coord: string
        Names of the x and y coordinates in `ds`.
    """
    def time_to_string(t):
        return time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(t.astype(int) / 1000000000))

    for t in ds.time:
        time_slice_xarray = ds.sel(time=t)
        export_slice_to_geotiff(time_slice_xarray,
                                path + "_" + time_to_string(t) + ".tif",
                                x_coord=x_coord, y_coord=y_coord)


def export_xarray_to_geotiff(data, tif_path, bands=None, no_data=-9999, crs="EPSG:4326",
                             x_coord='longitude', y_coord='latitude'):
    """
    Export a GeoTIFF from a 2D `xarray.Dataset`.

    Parameters
    ----------
    data: xarray.Dataset or xarray.DataArray
        An xarray with 2 dimensions to be exported as a GeoTIFF.
    tif_path: string
        The path to write the GeoTIFF file to. You should include the file extension.
    bands: list of string
        The bands to write - in the order they should be written.
        Ignored if `data` is an `xarray.DataArray`.
    no_data: int
        The nodata value.
    crs: string
        The CRS of the output.
    x_coord, y_coord: string
        The string names of the x and y dimensions.
    """
    if isinstance(data, xr.DataArray):
        height, width = data.sizes[y_coord], data.sizes[x_coord]
        count, dtype = 1, data.dtype
    else:
        if bands is None:
            bands = list(data.data_vars.keys())
        else:
            assrt_msg_begin = "The `data` parameter is an `xarray.Dataset`. "
            assert isinstance(bands, list), assrt_msg_begin + "Bands must be a list of strings."
            assert len(bands) > 0 and isinstance(bands[0], str), assrt_msg_begin + "You must supply at least one band."
        height, width = data.dims[y_coord], data.dims[x_coord]
        count, dtype = len(bands), data[bands[0]].dtype
    with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs=crs,
            transform=_get_transform_from_xr(data, x_coord=x_coord, y_coord=y_coord),
            nodata=no_data) as dst:
        if isinstance(data, xr.DataArray):
            dst.write(data.values, 1)
        else:
            for index, band in enumerate(bands):
                dst.write(data[band].values.astype(dtype), index + 1)
    dst.close()

## End export ##