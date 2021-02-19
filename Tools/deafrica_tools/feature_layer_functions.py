"""
This script exists simply to keep the notebooks
'Extract_training_data.ipynb' and 'Predict.ipynb' tidy by
not clutering them up with custom training data functions.
"""

import pyproj
import dask
import hdstats
import datacube
import numpy as np
import sys
import xarray as xr
import warnings
import dask.array as da
from datacube.utils.geometry import assign_crs
from datacube.testutils.io import rio_slurp_xarray
from odc.algo import randomize, reshape_for_geomedian, xr_reproject, xr_geomedian
from odc.algo._dask import reshape_yxbt

sys.path.append("../../Scripts")
from deafrica_bandindices import calculate_indices
from deafrica_temporal_statistics import xr_phenology, temporal_statistics
from deafrica_classificationtools import HiddenPrints
from deafrica_datahandling import load_ard

warnings.filterwarnings("ignore")


def gm_mads_two_seasons_predict(ds):
    dc = datacube.Datacube(app="training")
    ds = ds / 10_000
    ds1 = ds.sel(time=slice("2019-01", "2019-06"))
    ds2 = ds.sel(time=slice("2019-07", "2019-12"))

    def fun(ds, era):
        # geomedian and tmads
        # gm_mads = xr_geomedian_tmad(ds)
        gm_mads = xr_geomedian_tmad_new(ds).compute()
        gm_mads = calculate_indices(
            gm_mads,
            index=["NDVI", "LAI", "MNDWI"],
            drop=False,
            normalise=False,
            collection="s2",
        )

        gm_mads["sdev"] = -np.log(gm_mads["sdev"])
        gm_mads["bcdev"] = -np.log(gm_mads["bcdev"])
        gm_mads["edev"] = -np.log(gm_mads["edev"])
        gm_mads = gm_mads.chunk({"x": 2000, "y": 2000})

        # rainfall climatology
        if era == "_S1":
            chirps = assign_crs(
                xr.open_rasterio(
                    "/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc"
                ),
                crs="epsg:4326",
            )
        if era == "_S2":
            chirps = assign_crs(
                xr.open_rasterio(
                    "/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc"
                ),
                crs="epsg:4326",
            )

        chirps = xr_reproject(chirps, ds.geobox, "bilinear")
        chirps = chirps.chunk({"x": 2000, "y": 2000})
        gm_mads["rain"] = chirps

        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band: band + era})

        return gm_mads

    epoch1 = fun(ds1, era="_S1")
    epoch2 = fun(ds2, era="_S2")

    # slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name="slope").chunk({"x": 2000, "y": 2000})

    result = xr.merge([epoch1, epoch2, slope], compat="override")

    return result.squeeze()


def gm_mads_two_seasons_training(ds):
    dc = datacube.Datacube(app="training")
    ds = ds / 10000
    ds1 = ds.sel(time=slice("2019-01", "2019-06"))
    ds2 = ds.sel(time=slice("2019-07", "2019-12"))

    def fun(ds, era):
        # geomedian and tmads
        gm_mads = xr_geomedian_tmad(ds)
        gm_mads = calculate_indices(
            gm_mads,
            index=["NDVI", "LAI", "MNDWI"],
            drop=False,
            normalise=False,
            collection="s2",
        )

        gm_mads["sdev"] = -np.log(gm_mads["sdev"])
        gm_mads["bcdev"] = -np.log(gm_mads["bcdev"])
        gm_mads["edev"] = -np.log(gm_mads["edev"])

        # rainfall climatology
        if era == "_S1":
            chirps = assign_crs(
                xr.open_rasterio(
                    "/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc"
                ),
                crs="epsg:4326",
            )
        if era == "_S2":
            chirps = assign_crs(
                xr.open_rasterio(
                    "/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc"
                ),
                crs="epsg:4326",
            )

        chirps = xr_reproject(chirps, ds.geobox, "bilinear")
        gm_mads["rain"] = chirps

        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band: band + era})

        return gm_mads

    epoch1 = fun(ds1, era="_S1")
    epoch2 = fun(ds2, era="_S2")

    # slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name="slope")

    result = xr.merge([epoch1, epoch2, slope], compat="override")

    return result.squeeze()


def gm_mads_evi_rainfall(ds):
    """
    6 monthly and annual
    gm + mads
    evi stats (10, 50, 90 percentile, range, std)
    rainfall actual stats (min, mean, max, range, std) from monthly data
    rainfall clim stats (min, mean, max, range, std) from monthly data
    """
    dc = datacube.Datacube(app="training")
    ds = ds / 10000
    ds = ds.rename({"nir_1": "nir_wide", "nir_2": "nir"})
    ds1 = ds.sel(time=slice("2019-01", "2019-06"))
    ds2 = ds.sel(time=slice("2019-07", "2019-12"))

    chirps = []
    chpclim = []
    for m in range(1, 13):
        chirps.append(
            xr_reproject(
                assign_crs(
                    xr.open_rasterio(
                        f"/g/data/CHIRPS/monthly_2019/chirps-v2.0.2019.{m:02d}.tif"
                    )
                    .squeeze()
                    .expand_dims({"time": [m]}),
                    crs="epsg:4326",
                ),
                ds.geobox,
                "bilinear",
            )
        )
        chpclim.append(
            rio_slurp_xarray(
                f"https://deafrica-data-dev.s3.amazonaws.com/product-dev/deafrica_chpclim_50n_50s_{m:02d}.tif",
                gbox=ds.geobox,
                resapling="bilinear",
            ).expand_dims({"time": [m]})
        )

    chirps = xr.concat(chirps, dim="time")
    chpclim = xr.concat(chpclim, dim="time")

    def fun(ds, chirps, chpclim, era):
        ds = calculate_indices(
            ds, index=["EVI"], drop=False, normalise=False, collection="s2"
        )
        # geomedian and tmads
        gm_mads = xr_geomedian_tmad(ds)
        gm_mads = calculate_indices(
            gm_mads,
            index=["EVI", "NDVI", "LAI", "MNDWI"],
            drop=False,
            normalise=False,
            collection="s2",
        )

        gm_mads["sdev"] = -np.log(gm_mads["sdev"])
        gm_mads["bcdev"] = -np.log(gm_mads["bcdev"])
        gm_mads["edev"] = -np.log(gm_mads["edev"])

        # EVI stats
        gm_mads["evi_10"] = ds.EVI.quantile(0.1, dim="time")
        gm_mads["evi_50"] = ds.EVI.quantile(0.5, dim="time")
        gm_mads["evi_90"] = ds.EVI.quantile(0.9, dim="time")
        gm_mads["evi_range"] = gm_mads["evi_90"] - gm_mads["evi_10"]
        gm_mads["evi_std"] = ds.EVI.std(dim="time")

        # rainfall actual
        gm_mads["rain_min"] = chirps.min(dim="time")
        gm_mads["rain_mean"] = chirps.mean(dim="time")
        gm_mads["rain_max"] = chirps.max(dim="time")
        gm_mads["rain_range"] = gm_mads["rain_max"] - gm_mads["rain_min"]
        gm_mads["rain_std"] = chirps.std(dim="time")

        # rainfall climatology
        gm_mads["rainclim_min"] = chpclim.min(dim="time")
        gm_mads["rainclim_mean"] = chpclim.mean(dim="time")
        gm_mads["rainclim_max"] = chpclim.max(dim="time")
        gm_mads["rainclim_range"] = gm_mads["rainclim_max"] - gm_mads["rainclim_min"]
        gm_mads["rainclim_std"] = chpclim.std(dim="time")

        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band: band + era})

        return gm_mads

    epoch0 = fun(ds, chirps, chpclim, era="_S0")
    time, month = slice("2019-01", "2019-06"), slice(1, 6)
    epoch1 = fun(
        ds.sel(time=time), chirps.sel(time=month), chpclim.sel(time=month), era="_S1"
    )
    time, month = slice("2019-07", "2019-12"), slice(7, 12)
    epoch2 = fun(
        ds.sel(time=time), chirps.sel(time=month), chpclim.sel(time=month), era="_S2"
    )

    # slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name="slope")

    result = xr.merge([epoch0, epoch1, epoch2, slope], compat="override")

    return result.squeeze()


def xr_geomedian_tmad(ds, axis="time", where=None, **kw):
    """
    :param ds: xr.Dataset|xr.DataArray|numpy array
    Other parameters:
    **kwargs -- passed on to pcm.gnmpcm
       maxiters   : int         1000
       eps        : float       0.0001
       num_threads: int| None   None
    """

    import hdstats

    def gm_tmad(arr, **kw):
        """
        arr: a high dimensional numpy array where the last dimension will be reduced.

        returns: a numpy array with one less dimension than input.
        """
        gm = hdstats.nangeomedian_pcm(arr, **kw)
        nt = kw.pop("num_threads", None)
        emad = hdstats.emad_pcm(arr, gm, num_threads=nt)[:, :, np.newaxis]
        smad = hdstats.smad_pcm(arr, gm, num_threads=nt)[:, :, np.newaxis]
        bcmad = hdstats.bcmad_pcm(arr, gm, num_threads=nt)[:, :, np.newaxis]
        return np.concatenate([gm, emad, smad, bcmad], axis=-1)

    def norm_input(ds, axis):
        if isinstance(ds, xr.DataArray):
            xx = ds
            if len(xx.dims) != 4:
                raise ValueError("Expect 4 dimensions on input: y,x,band,time")
            if axis is not None and xx.dims[3] != axis:
                raise ValueError(
                    f"Can only reduce last dimension, expect: y,x,band,{axis}"
                )
            return None, xx, xx.data
        elif isinstance(ds, xr.Dataset):
            xx = reshape_for_geomedian(ds, axis)
            return ds, xx, xx.data
        else:  # assume numpy or similar
            xx_data = ds
            if xx_data.ndim != 4:
                raise ValueError("Expect 4 dimensions on input: y,x,band,time")
            return None, None, xx_data

    kw.setdefault("nocheck", False)
    kw.setdefault("num_threads", 1)
    kw.setdefault("eps", 1e-6)

    ds, xx, xx_data = norm_input(ds, axis)
    is_dask = dask.is_dask_collection(xx_data)

    if where is not None:
        if is_dask:
            raise NotImplementedError(
                "Dask version doesn't support output masking currently"
            )

        if where.shape != xx_data.shape[:2]:
            raise ValueError("Shape for `where` parameter doesn't match")
        set_nan = ~where
    else:
        set_nan = None

    if is_dask:
        if xx_data.shape[-2:] != xx_data.chunksize[-2:]:
            xx_data = xx_data.rechunk(xx_data.chunksize[:2] + (-1, -1))

        data = da.map_blocks(
            lambda x: gm_tmad(x, **kw),
            xx_data,
            name=randomize("geomedian"),
            dtype=xx_data.dtype,
            chunks=xx_data.chunks[:-2] + (xx_data.chunks[-2][0] + 3,),
            drop_axis=3,
        )
    else:
        data = gm_tmad(xx_data, **kw)

    if set_nan is not None:
        data[set_nan, :] = np.nan

    if xx is None:
        return data

    dims = xx.dims[:-1]
    cc = {k: xx.coords[k] for k in dims}
    cc[dims[-1]] = np.hstack([xx.coords[dims[-1]].values, ["edev", "sdev", "bcdev"]])
    xx_out = xr.DataArray(data, dims=dims, coords=cc)

    if ds is None:
        xx_out.attrs.update(xx.attrs)
        return xx_out

    ds_out = xx_out.to_dataset(dim="band")
    for b in ds.data_vars.keys():
        src, dst = ds[b], ds_out[b]
        dst.attrs.update(src.attrs)

    return assign_crs(ds_out, crs=ds.geobox.crs)


def xr_geomedian_tmad_new(ds, **kw):
    """
    Same as other one but uses reshape_yxbt instead of
    reshape_for_geomedian
    """

    import hdstats

    def gm_tmad(arr, **kw):
        """
        arr: a high dimensional numpy array where the last dimension will be reduced.

        returns: a numpy array with one less dimension than input.
        """
        gm = hdstats.nangeomedian_pcm(arr, **kw)
        nt = kw.pop("num_threads", None)
        emad = hdstats.emad_pcm(arr, gm, num_threads=nt)[:, :, np.newaxis]
        smad = hdstats.smad_pcm(arr, gm, num_threads=nt)[:, :, np.newaxis]
        bcmad = hdstats.bcmad_pcm(arr, gm, num_threads=nt)[:, :, np.newaxis]
        return np.concatenate([gm, emad, smad, bcmad], axis=-1)

    def norm_input(ds):
        if isinstance(ds, xr.Dataset):
            xx = reshape_yxbt(ds, yx_chunks=500)
            return ds, xx, xx.data

    kw.setdefault("nocheck", False)
    kw.setdefault("num_threads", 1)
    kw.setdefault("eps", 1e-6)

    ds, xx, xx_data = norm_input(ds)
    is_dask = dask.is_dask_collection(xx_data)

    if is_dask:
        data = da.map_blocks(
            lambda x: gm_tmad(x, **kw),
            xx_data,
            name=randomize("geomedian"),
            dtype=xx_data.dtype,
            chunks=xx_data.chunks[:-2] + (xx_data.chunks[-2][0] + 3,),
            drop_axis=3,
        )

    dims = xx.dims[:-1]
    cc = {k: xx.coords[k] for k in dims}
    cc[dims[-1]] = np.hstack([xx.coords[dims[-1]].values, ["edev", "sdev", "bcdev"]])
    xx_out = xr.DataArray(data, dims=dims, coords=cc)

    if ds is None:
        xx_out.attrs.update(xx.attrs)
        return xx_out

    ds_out = xx_out.to_dataset(dim="band")
    for b in ds.data_vars.keys():
        src, dst = ds[b], ds_out[b]
        dst.attrs.update(src.attrs)

    return assign_crs(ds_out, crs=ds.geobox.crs)
