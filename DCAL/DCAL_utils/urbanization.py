import numpy as np

from vegetation import NDVI

def NDBI(ds):
    """
    Computes the Normalized Difference Built-up Index for an `xarray.Dataset`.
    The formula is (SWIR1 - NIR) / (SWIR1 + NIR).
    Values should be in the range [-1,1] for valid LANDSAT data (swir1 and nir are positive).

    This is a spectral index for which high values often indicate urban areas.
    Note that DBSI often performs better in arid and semi-arid environments, since
    NDBI does not differentiate bare soil from urban areas well.

    Parameters
    ----------
    ds: xarray.Dataset
        An `xarray.Dataset` that must contain 'swir1' and 'nir' `DataArrays`.

    Returns
    -------
    ndbi: xarray.DataArray
        An `xarray.DataArray` with the same shape as `ds` - the same coordinates in
        the same order.y
    """
    return (ds.swir1 - ds.nir) / (ds.swir1 + ds.nir)


def DBSI(ds, normalize=True):
    """
    Computes the Dry Bare-Soil Index as defined in the paper "Applying
    Built-Up and Bare-Soil Indices from Landsat 8 to Cities in Dry Climates".
    The formula is (SWIR1 - GREEN) / (SWIR1 + GREEN) - NDVI.
    If `normalize == False`, returned values should be in the range [-2,2].

    This is a spectral index for which high values often indicate bare soil and
    low values often indicate urban areas.
    Note that DBSI often performs better in arid and semi-arid environments than NDBI, since
    it differentiates bare soil from urban areas better.

    Parameters
    ----------
    ds: xarray.Dataset
        An `xarray.Dataset` that must contain
        'swir1', 'green', 'nir', and 'red' `DataArrays`.
    normalize: boolean
        Whether to normalize to the range [-1,1] - the range of most common spectral indices.

    Returns
    -------
    dbsi: xarray.DataArray
        An `xarray.DataArray` with the same shape as `ds` - the same coordinates in
        the same order.
    """
    dbsi = (ds.swir1 - ds.green) / (ds.swir1 + ds.green) - NDVI(ds)
    if normalize:
        dbsi.values = np.interp(dbsi.values, (-2, 2), (-1, 1))
    return dbsi