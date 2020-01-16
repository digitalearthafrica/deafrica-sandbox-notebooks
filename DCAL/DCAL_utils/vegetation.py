import numpy as np

def EVI(ds, G=2.5, C1=6, C2=7.5, L=1, normalize=True):
    """
    Computes the 3-band Enhanced Vegetation Index for an `xarray.Dataset`.
    The formula is G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L).
    Usually, G = 2.5, C1 = 6, C2 = 7.5, and L = 1.
    For Landsat data, returned values should be in the range [-1,1] if `normalize == True`.
    If `normalize == False`, returned values should be in the range [-1,2.5].

    EVI is superior to NDVI in accuracy because it is less dependent on the solar
    incidence angle, atmospheric conditions (e.g. particles and clouds), shadows, and
    soil appearance.

    Parameters
    ----------
    ds: xarray.Dataset
        An `xarray.Dataset` that must contain 'nir', 'red', and 'blue' `DataArrays`.
    G, C1, C2, L: float
        G is the gain factor - a constant scaling factor.
        C1 and C2 pertain to aerosols in clouds.
        L adjusts for canopy background and soil appearance. It particularly pertains to
        the nir and red bands, which are transmitted non-linearly through a canopy.
    normalize: boolean
        Whether to normalize to the range [-1,1] - the range of most common spectral indices.

    Returns
    -------
    evi: xarray.DataArray
        An `xarray.DataArray` with the same shape as `ds` - the same coordinates in
        the same order.
    """
    evi = G * (ds.nir - ds.red) / (ds.nir + C1 * ds.red - C2 * ds.blue + L)
    # Clamp values to the range [-1,2.5].
    evi.values[evi.values < -1] = -1
    evi.values[2.5 < evi.values] = 2.5
    if normalize:
        # Scale values in the  range [0,2.5] to the range [0,1].
        pos_vals_mask = 0 < evi.values
        evi.values[pos_vals_mask] = np.interp(evi.values[pos_vals_mask], (0, 2.5), (0, 1))
    return evi


def EVI2(ds, G=2.5, C=2.4, L=1, normalize=True):
    """
    Computes the 2-band Enhanced Vegetation Index for an `xarray.Dataset`.
    The formula is G*((NIR-RED)/(NIR+C*Red+L)).
    Usually, G = 2.5, C = 2.4, and L = 1.
    For Landsat data, returned values should be in the range [-1,1] if `normalize == True`.
    If `normalize == False`, returned values should be in the range [-1,2.5].

    EVI2 does not require a blue band like EVI, which means less data is required to use it.
    Additionally, the blue band used in EVI can have a low signal-to-noise ratio
    in earth observation imagery. When atmospheric effects are insignificant (e.g. on clear days),
    EVI2 should closely match EVI.

    Parameters
    ----------
    ds: xarray.Dataset
        An `xarray.Dataset` that must contain 'nir', and 'red' `DataArrays`.
    G, C, L: float
        G is the gain factor - a constant scaling factor.
        C pertains to aerosols in clouds.
        L adjusts for canopy background and soil appearance. It particularly pertains to
        the nir and red bands, which are transmitted non-linearly through a canopy.
    normalize: boolean
        Whether to normalize to the range [-1,1] - the range of most common spectral indices.

    Returns
    -------
    evi: xarray.DataArray
        An `xarray.DataArray` with the same shape as `ds` - the same coordinates in
        the same order.
    """
    evi = G * (ds.nir - ds.red) / (ds.nir + C * ds.red + L)
    # Clamp values to the range [-1,2.5].
    evi.values[evi.values < -1] = -1
    evi.values[2.5 < evi.values] = 2.5
    if normalize:
        # Scale values in the  range [0,2.5] to the range [0,1].
        pos_vals_mask = 0 < evi.values
        evi.values[pos_vals_mask] = np.interp(evi.values[pos_vals_mask], (0, 2.5), (0, 1))
    return evi

def NBR(ds):
    """
    Computes the Normalized Burn Ratio for an `xarray.Dataset`.
    The formula is (NIR - SWIR2) / (NIR + SWIR2).
    Values should be in the range [-1,1] for valid LANDSAT data (nir and swir2 are positive).

    Parameters
    ----------
    ds: xarray.Dataset
        An `xarray.Dataset` that must contain 'nir' and 'swir2' `DataArrays`.

    Returns
    -------
    nbr: xarray.DataArray
        An `xarray.DataArray` with the same shape as `ds` - the same coordinates in
        the same order.
    """
    return (ds.nir - ds.swir2) / (ds.nir + ds.swir2)

def NDVI(ds):
    """
    Computes the Normalized Difference Vegetation Index for an `xarray.Dataset`.
    The formula is (NIR - RED) / (NIR + RED).
    Values should be in the range [-1,1] for valid LANDSAT data (nir and red are positive).

    Parameters
    ----------
    ds: xarray.Dataset
        An `xarray.Dataset` that must contain 'nir' and 'red' `DataArrays`.

    Returns
    -------
    ndvi: xarray.DataArray
        An `xarray.DataArray` with the same shape as `ds` - the same coordinates in
        the same order.
    """
    return (ds.nir - ds.red) / (ds.nir + ds.red)


def SAVI(ds, L=0.5, normalize=True):
    """
    Computes the Soil-Adjusted Vegetation Index for an `xarray.Dataset`.
    The formula is (NIR - RED) / (NIR + RED + L) * (1 + L).
    For Landsat data, returned values should be in the range [-1,1] if `normalize == True`.
    If `normalize == False`, returned values should be in the range [-1-L,1+L].

    In areas where vegetative cover is low (i.e., < 40%) and the soil surface
    is exposed, the reflectance of light in the red and near-infrared spectra
    can influence vegetation index values. This is especially problematic when
    comparisons are being made across different soil types that may reflect different
    amounts of light in the red and near infrared wavelengths (i.e. soils with
    different brightness values). The soil-adjusted vegetation index was developed
    as a modification of the Normalized Difference Vegetation Index to correct for
    the influence of soil brightness when vegetative cover is low.

    Parameters
    ----------
    ds: xarray.Dataset
        An `xarray.Dataset` that must contain 'nir', and 'red' `DataArrays`.
    L: float
        L is the “soil brightness correction factor”, which should be varied based
        on the greenness of vegetation in the scene. In very high vegetation regions,
        `L=0`. In areas with no green vegetation, `L=1`. Generally, `L=0.5` works well
        and is the default value. When `L=0`, `SAVI==NDVI`.
    normalize: boolean
        Whether to normalize to the range [-1,1] - the range of most common spectral indices.

    Returns
    -------
    savi: xarray.DataArray
        An `xarray.DataArray` with the same shape as `ds` - the same coordinates in
        the same order.
    """
    savi = (ds.nir - ds.red) / (ds.nir + ds.red + L) * (1 + L)
    if normalize:
        savi.values = np.interp(savi.values, (-1-L, 1+L), (-1, 1))
    return savi