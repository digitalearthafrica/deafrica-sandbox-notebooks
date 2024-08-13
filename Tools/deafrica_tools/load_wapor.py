import collections

import numpy as np
import pandas as pd
import requests
import rioxarray
import xarray as xr

from deafrica_tools.spatial import add_geobox

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

    return output_df


def load_wapor_ds(filename: str, variable: str) -> xr.Dataset:
    """
    Load the netCDF downloaded using `wapordl.wapor_map` as a
    xarray Dataset. 
    Note: Only works for netcdf files, this is because when loading the WAPOR TIFF 
    files, the start_date and end_date attributes are only loaded for the first band.

    Parameters
    ----------
    filename : str
        File path of the netCDF downloaded using `wapordl.wapor_map`
    variable : str
        Name of the WAPOR version 3 mapset downloaded

    Returns
    -------
    xr.Dataset
        Dataset containing the WAPOR version 3 mapset data downloaded.
    """
    # Load the file.
    if filename.endswith(".tif"):
        raise ValueError("Please set extension='.nc' when downloading data using `wapor_map`")
    elif filename.endswith(".nc"):
        ds = rioxarray.open_rasterio(filename).squeeze(dim="band").drop_vars("band")
        
    # Store the crs in the spatial_ref coordinate
    if "spatial_ref" in ds.coords:
        crs_attrs = ds["spatial_ref"].attrs
        ds = add_geobox(ds.drop_vars("spatial_ref"), crs=ds.rio.crs)
    elif "crs" in ds.coords:
        crs_attrs = ds["crs"].attrs
        ds = add_geobox(ds.drop_vars("crs"), crs=ds.rio.crs)
    elif "transverse_mercator" in ds.coords:
        crs_attrs = ds["transverse_mercator"].attrs
        ds = add_geobox(ds.drop_vars("transverse_mercator"), crs=ds.rio.crs)

    ds["spatial_ref"].attrs.update(crs_attrs)

    # Get the bands names.
    bands = [var for var in list(ds.variables) if var not in list(ds.coords)]
    # Assign the correct time coordinate to each band.
    # This is the start date in the attributes.
    da_list = [
        ds[band]
        .assign_coords(time=np.datetime64(ds[band].attrs["start_date"], "ns"))
        .expand_dims(time=1)
        .rename(variable)
        for band in bands
    ]

    # Merge the DataArrays
    da = xr.concat(da_list, dim="time")
    
    # Rescale data
    xr.set_options(keep_attrs=True)
    if "scale_factor" in da.attrs:
        da = da * da.attrs['scale_factor'] + da.attrs['add_offset']

    # Edit the attributes
    if len(da.time) > 1:
        attrs_to_drop = ["start_date", "end_date", "number_of_days"]
        for attr in attrs_to_drop:
            del da.attrs[attr]

    return da.to_dataset(promote_attrs=True).sortby("time", ascending=True)
