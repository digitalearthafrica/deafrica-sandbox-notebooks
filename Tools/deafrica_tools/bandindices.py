"""
Functions for computing remote sensing band indices on Digital Earth Africa
data.

.. autosummary::
   :nosignatures:
   :toctree: gen

"""

# Import required packages
import warnings
import numpy as np

# Define custom functions
def calculate_indices(
    ds,
    index=None,
    collection=None,
    custom_varname=None,
    normalise=True,
    drop=False,
    deep_copy=True,
):
    """
    Takes an xarray dataset containing spectral bands, calculates one of
    a set of remote sensing indices, and adds the resulting array as a
    new variable in the original dataset.

    Last modified: July 2021

    Parameters
    ----------
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array with containing the
        spectral bands required to calculate the index. These bands are
        used as inputs to calculate the selected water index.
    index : str or list of strs
        A string giving the name of the index to calculate or a list of
        strings giving the names of the indices to calculate:

        * ``'AWEI_ns'`` (Automated Water Extraction Index, no shadows, Feyisa 2014)
        * ``'AWEI_sh'`` (Automated Water Extraction Index, shadows, Feyisa 2014)
        * ``'BAEI'`` (Built-Up Area Extraction Index, Bouzekri et al. 2015)
        * ``'BAI'`` (Burn Area Index, Martin 1998)
        * ``'BSI'`` (Bare Soil Index, Rikimaru et al. 2002)
        * ``'BUI'`` (Built-Up Index, He et al. 2010)
        * ``'CMR'`` (Clay Minerals Ratio, Drury 1987)
        * ``'ENDISI'`` (Enhanced Normalised Difference for Impervious Surfaces Index, Chen et al. 2019)
        * ``'EVI'`` (Enhanced Vegetation Index, Huete 2002)
        * ``'FMR'`` (Ferrous Minerals Ratio, Segal 1982)
        * ``'IOR'`` (Iron Oxide Ratio, Segal 1982)
        * ``'LAI'`` (Leaf Area Index, Boegh 2002)
        * ``'MNDWI'`` (Modified Normalised Difference Water Index, Xu 1996)
        * ``'MSAVI'`` (Modified Soil Adjusted Vegetation Index, Qi et al. 1994)
        * ``'NBI'`` (New Built-Up Index, Jieli et al. 2010)
        * ``'NBR'`` (Normalised Burn Ratio, Lopez Garcia 1991)
        * ``'NDBI'`` (Normalised Difference Built-Up Index, Zha 2003)
        * ``'NDCI'`` (Normalised Difference Chlorophyll Index, Mishra & Mishra, 2012)
        * ``'NDMI'`` (Normalised Difference Moisture Index, Gao 1996)
        * ``'NDSI'`` (Normalised Difference Snow Index, Hall 1995)
        * ``'NDVI'`` (Normalised Difference Vegetation Index, Rouse 1973)
        * ``'NDWI'`` (Normalised Difference Water Index, McFeeters 1996)
        * ``'SAVI'`` (Soil Adjusted Vegetation Index, Huete 1988)
        * ``'TCB'`` (Tasseled Cap Brightness, Crist 1985)
        * ``'TCG'`` (Tasseled Cap Greeness, Crist 1985)
        * ``'TCW'`` (Tasseled Cap Wetness, Crist 1985)
        * ``'WI'`` (Water Index, Fisher 2016)

    collection : str
        An string that tells the function what data collection is
        being used to calculate the index. This is necessary because
        different collections use different names for bands covering
        a similar spectra.

        Valid options are:

         * ``'c2'`` (for USGS Landsat Collection 2)
         * ``'s2'`` (for Sentinel-2)
         As of July 2021, options for ``'c1'`` (USGS Landsat Collection 1)
         have been removed as Collection 1 data has been archived. The 
         improved version of Landsat data can be accessed through Collection 2.

    custom_varname : str, optional
        By default, the original dataset will be returned with
        a new index variable named after `index` (e.g. 'NDVI'). To
        specify a custom name instead, you can supply e.g.
        `custom_varname='custom_name'`. Defaults to None, which uses
        `index` to name the variable.
    normalise : bool, optional
        Some coefficient-based indices (e.g. ``'WI'``, ``'BAEI'``,
        ``'AWEI_ns'``, ``'AWEI_sh'``, ``'TCW'``, ``'TCG'``, ``'TCB'``,
        ``'EVI'``, ``'LAI'``, ``'SAVI'``, ``'MSAVI'``)
        produce different results if surface reflectance values are not
        scaled between 0.0 and 1.0 prior to calculating the index.
        Setting `normalise=True` first scales values to a 0.0-1.0 range
        by dividing by 10000.0. Defaults to True.
    drop : bool, optional
        Provides the option to drop the original input data, thus saving
        space. If `drop=True`, returns only the index and its values.
    deep_copy: bool, optional
        If `deep_copy=False`, calculate_indices will modify the original
        array, adding bands to the input dataset and not removing them.
        If the calculate_indices function is run more than once, variables
        may be dropped incorrectly producing unexpected behaviour. This is
        a bug and may be fixed in future releases. This is only a problem
        when `drop=True`.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a
        new varible containing the remote sensing index as a DataArray.
        If drop = True, the new variable/s as DataArrays in the
        original Dataset.
    """

    # Set ds equal to a copy of itself in order to prevent the function
    # from editing the input dataset. This is to prevent unexpected
    # behaviour though it uses twice as much memory.
    if deep_copy:
        ds = ds.copy(deep=True)

    # Capture input band names in order to drop these if drop=True
    if drop:
        bands_to_drop = list(ds.data_vars)
        print(f"Dropping bands {bands_to_drop}")

    # Dictionary containing remote sensing index band recipes
    index_dict = {
        # Normalised Difference Vegation Index, Rouse 1973
        "NDVI": lambda ds: (ds.nir - ds.red) / (ds.nir + ds.red),
        # Enhanced Vegetation Index, Huete 2002
        "EVI": lambda ds: (
            (2.5 * (ds.nir - ds.red)) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1)
        ),
        # Leaf Area Index, Boegh 2002
        "LAI": lambda ds: (
            3.618
            * ((2.5 * (ds.nir - ds.red)) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))
            - 0.118
        ),
        # Soil Adjusted Vegetation Index, Huete 1988
        "SAVI": lambda ds: ((1.5 * (ds.nir - ds.red)) / (ds.nir + ds.red + 0.5)),
        # Mod. Soil Adjusted Vegetation Index, Qi et al. 1994
        "MSAVI": lambda ds: (
            (2 * ds.nir + 1 - ((2 * ds.nir + 1) ** 2 - 8 * (ds.nir - ds.red)) ** 0.5)
            / 2
        ),
        # Normalised Difference Moisture Index, Gao 1996
        "NDMI": lambda ds: (ds.nir - ds.swir_1) / (ds.nir + ds.swir_1),
        # Normalised Burn Ratio, Lopez Garcia 1991
        "NBR": lambda ds: (ds.nir - ds.swir_2) / (ds.nir + ds.swir_2),
        # Burn Area Index, Martin 1998
        "BAI": lambda ds: (1.0 / ((0.10 - ds.red) ** 2 + (0.06 - ds.nir) ** 2)),
        # Normalised Difference Chlorophyll Index,
        # (Mishra & Mishra, 2012)
        "NDCI": lambda ds: (ds.red_edge_1 - ds.red) / (ds.red_edge_1 + ds.red),
        # Normalised Difference Snow Index, Hall 1995
        "NDSI": lambda ds: (ds.green - ds.swir_1) / (ds.green + ds.swir_1),
        # Normalised Difference Water Index, McFeeters 1996
        "NDWI": lambda ds: (ds.green - ds.nir) / (ds.green + ds.nir),
        # Modified Normalised Difference Water Index, Xu 2006
        "MNDWI": lambda ds: (ds.green - ds.swir_1) / (ds.green + ds.swir_1),
        # Normalised Difference Built-Up Index, Zha 2003
        "NDBI": lambda ds: (ds.swir_1 - ds.nir) / (ds.swir_1 + ds.nir),
        # Built-Up Index, He et al. 2010
        "BUI": lambda ds: ((ds.swir_1 - ds.nir) / (ds.swir_1 + ds.nir))
        - ((ds.nir - ds.red) / (ds.nir + ds.red)),
        # Built-up Area Extraction Index, Bouzekri et al. 2015
        "BAEI": lambda ds: (ds.red + 0.3) / (ds.green + ds.swir_1),
        # New Built-up Index, Jieli et al. 2010
        "NBI": lambda ds: (ds.swir_1 + ds.red) / ds.nir,
        # Bare Soil Index, Rikimaru et al. 2002
        "BSI": lambda ds: ((ds.swir_1 + ds.red) - (ds.nir + ds.blue))
        / ((ds.swir_1 + ds.red) + (ds.nir + ds.blue)),
        # Automated Water Extraction Index (no shadows), Feyisa 2014
        "AWEI_ns": lambda ds: (
            4 * (ds.green - ds.swir_1) - (0.25 * ds.nir * +2.75 * ds.swir_2)
        ),
        # Automated Water Extraction Index (shadows), Feyisa 2014
        "AWEI_sh": lambda ds: (
            ds.blue + 2.5 * ds.green - 1.5 * (ds.nir + ds.swir_1) - 0.25 * ds.swir_2
        ),
        # Water Index, Fisher 2016
        "WI": lambda ds: (
            1.7204
            + 171 * ds.green
            + 3 * ds.red
            - 70 * ds.nir
            - 45 * ds.swir_1
            - 71 * ds.swir_2
        ),
        # Tasseled Cap Wetness, Crist 1985
        "TCW": lambda ds: (
            0.0315 * ds.blue
            + 0.2021 * ds.green
            + 0.3102 * ds.red
            + 0.1594 * ds.nir
            + -0.6806 * ds.swir_1
            + -0.6109 * ds.swir_2
        ),
        # Tasseled Cap Greeness, Crist 1985
        "TCG": lambda ds: (
            -0.1603 * ds.blue
            + -0.2819 * ds.green
            + -0.4934 * ds.red
            + 0.7940 * ds.nir
            + -0.0002 * ds.swir_1
            + -0.1446 * ds.swir_2
        ),
        # Tasseled Cap Brightness, Crist 1985
        "TCB": lambda ds: (
            0.2043 * ds.blue
            + 0.4158 * ds.green
            + 0.5524 * ds.red
            + 0.5741 * ds.nir
            + 0.3124 * ds.swir_1
            + -0.2303 * ds.swir_2
        ),
        # Clay Minerals Ratio, Drury 1987
        "CMR": lambda ds: (ds.swir_1 / ds.swir_2),
        # Ferrous Minerals Ratio, Segal 1982
        "FMR": lambda ds: (ds.swir_1 / ds.nir),
        # Iron Oxide Ratio, Segal 1982
        "IOR": lambda ds: (ds.red / ds.blue),
    }
    
    # Enhanced Normalised Difference Impervious Surfaces Index, Chen et al. 2019
    def mndwi(ds):
        return (ds.green - ds.swir_1) / (ds.green + ds.swir_1)
    def swir_diff(ds):
        return ds.swir_1/ds.swir_2
    def alpha(ds):
        return (2*(np.mean(ds.blue)))/(np.mean(swir_diff(ds)) + np.mean(mndwi(ds)**2))
    def ENDISI(ds):
        m = mndwi(ds)
        s = swir_diff(ds)
        a = alpha(ds)
        return (ds.blue - (a)*(s + m**2))/(ds.blue + (a)*(s + m**2))
    
    index_dict["ENDISI"] = ENDISI

    # If index supplied is not a list, convert to list. This allows us to
    # iterate through either multiple or single indices in the loop below
    indices = index if isinstance(index, list) else [index]

    # calculate for each index in the list of indices supplied (indexes)
    for index in indices:

        # Select an index function from the dictionary
        index_func = index_dict.get(str(index))

        # If no index is provided or if no function is returned due to an
        # invalid option being provided, raise an exception informing user to
        # choose from the list of valid options
        if index is None:

            raise ValueError(
                f"No remote sensing `index` was provided. Please "
                "refer to the function \ndocumentation for a full "
                "list of valid options for `index` (e.g. 'NDVI')"
            )

        elif (
            index
            in [
                "WI",
                "BAEI",
                "AWEI_ns",
                "AWEI_sh",
                "EVI",
                "LAI",
                "SAVI",
                "MSAVI",
            ]
            and not normalise
        ):

            warnings.warn(
                f"\nA coefficient-based index ('{index}') normally "
                "applied to surface reflectance values in the \n"
                "0.0-1.0 range was applied to values in the 0-10000 "
                "range. This can produce unexpected results; \nif "
                "required, resolve this by setting `normalise=True`"
            )

        elif index_func is None:

            raise ValueError(
                f"The selected index '{index}' is not one of the "
                "valid remote sensing index options. \nPlease "
                "refer to the function documentation for a full "
                "list of valid options for `index`"
            )

        # Rename bands to a consistent format if depending on what collection
        # is specified in `collection`. This allows the same index calculations
        # to be applied to all collections. If no collection was provided,
        # raise an exception.
        if collection is None:

            raise ValueError(
                "No `collection` was provided. Please specify "
                "either 'c2' or 's2' to ensure the \nfunction "
                "calculates indices using the correct spectral "
                "bands"
            )
            
        elif collection == "c2":
            sr_max = 1.0
            # Dictionary mapping full data names to simpler alias names
            # This only applies to properly-scaled C2 data i.e. from
            # the Landsat geomedians. calculate_indices will not show 
            # correct output for raw (unscaled) Landsat data (i.e. default
            # outputs from dc.load)
            bandnames_dict = {
                "SR_B1": "blue",
                "SR_B2": "green",
                "SR_B3": "red",
                "SR_B4": "nir",
                "SR_B5": "swir_1",
                "SR_B7": "swir_2",
                }
            
            # Rename bands in dataset to use simple names (e.g. 'red')
            bands_to_rename = {
                a: b for a, b in bandnames_dict.items() if a in ds.variables
            }

        elif collection == "s2":
            sr_max = 10000
            # Dictionary mapping full data names to simpler alias names
            bandnames_dict = {
                "nir_1": "nir",
                "B02": "blue",
                "B03": "green",
                "B04": "red",
                "B05": "red_edge_1",
                "B06": "red_edge_2",
                "B07": "red_edge_3",
                "B08": "nir",
                "B11": "swir_1",
                "B12": "swir_2",
                }

            # Rename bands in dataset to use simple names (e.g. 'red')
            bands_to_rename = {
                a: b for a, b in bandnames_dict.items() if a in ds.variables
            }

        # Raise error if no valid collection name is provided:
        else:
            raise ValueError(
                f"'{collection}' is not a valid option for "
                "`collection`. Please specify either \n"
                "'c2' or 's2'"
            )

        # Apply index function
        try:
            # If normalised=True, divide data by 10,000 before applying func
            mult = sr_max if normalise else 1.0
            index_array = index_func(ds.rename(bands_to_rename) / mult)

        except AttributeError:
            raise ValueError(
                f"Please verify that all bands required to "
                f"compute {index} are present in `ds`."
            )

        # Add as a new variable in dataset
        output_band_name = custom_varname if custom_varname else index
        ds[output_band_name] = index_array

    # Once all indexes are calculated, drop input bands if drop=True
    if drop:
        ds = ds.drop(bands_to_drop)

    # Return input dataset with added water index variable
    return ds

def dualpol_indices(
    ds,
    co_pol='vv',
    cross_pol='vh',
    index=None,
    custom_varname=None,
    drop=False,
    deep_copy=True,
):
    """
    Takes an xarray dataset containing dual-polarization radar backscatter,
    calculates one or a set of indices, and adds the resulting array as a
    new variable in the original dataset.

    Last modified: July 2021

    Parameters
    ----------
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array containing the
        two polarization bands.

    co_pol: str
        Measurement name for the co-polarization band.
        Default is 'vv' for Sentinel-1.

    cross_pol: str
        Measurement name for the cross-polarization band.
        Default is 'vh' for Sentinel-1.

    index : str or list of strs
        A string giving the name of the index to calculate or a list of
        strings giving the names of the indices to calculate:

        * ``'RVI'`` (Radar Vegetation Index for dual-pol, Trudel et al. 2012; Nasirzadehdizaji et al., 2019; Gururaj et al., 2019)
        * ``'VDDPI'`` (Vertical dual depolarization index, Periasamy 2018)
        * ``'theta'`` (pseudo scattering-type, Bhogapurapu et al. 2021)
        * ``'entropy'`` (pseudo scattering entropy, Bhogapurapu et al. 2021)
        * ``'purity'`` (co-pol purity, Bhogapurapu et al. 2021)
        * ``'ratio'`` (cross-pol/co-pol ratio)

    custom_varname : str, optional
        By default, the original dataset will be returned with
        a new index variable named after `index` (e.g. 'RVI'). To
        specify a custom name instead, you can supply e.g.
        `custom_varname='custom_name'`. Defaults to None, which uses
        `index` to name the variable.

    drop : bool, optional
        Provides the option to drop the original input data, thus saving
        space. If `drop=True`, returns only the index and its values.

    deep_copy: bool, optional
        If `deep_copy=False`, calculate_indices will modify the original
        array, adding bands to the input dataset and not removing them.
        If the calculate_indices function is run more than once, variables
        may be dropped incorrectly producing unexpected behaviour. This is
        a bug and may be fixed in future releases. This is only a problem
        when `drop=True`.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a
        new varible containing the remote sensing index as a DataArray.
        If drop = True, the new variable/s as DataArrays in the
        original Dataset.
    """

    if not co_pol in list(ds.data_vars):
        raise ValueError(f"{co_pol} measurement is not in the dataset")
    if not cross_pol in list(ds.data_vars):
        raise ValueError(f"{cross_pol} measurement is not in the dataset")

    # Set ds equal to a copy of itself in order to prevent the function
    # from editing the input dataset. This is to prevent unexpected
    # behaviour though it uses twice as much memory.
    if deep_copy:
        ds = ds.copy(deep=True)

    # Capture input band names in order to drop these if drop=True
    if drop:
        bands_to_drop = list(ds.data_vars)
        print(f"Dropping bands {bands_to_drop}")

    def ratio(ds):
        return ds[cross_pol] / ds[co_pol]

    def purity(ds):
        return (1 - ratio(ds)) / (1 + ratio(ds))

    def theta(ds):
        return np.arctan((1 - ratio(ds))**2 / (1 + ratio(ds)**2 - ratio(ds)))

    def P1(ds):
        return 1 / (1 + ratio(ds))

    def P2(ds):
        return 1 - P1(ds)

    def entropy(ds):
        return P1(ds)*np.log2(P1(ds)) + P2(ds)*np.log2(P2(ds))

    # Dictionary containing remote sensing index band recipes
    index_dict = {
        # Radar Vegetation Index for dual-pol, Trudel et al. 2012
        "RVI": lambda ds: 4*ds[cross_pol] / (ds[co_pol] + ds[cross_pol]),
        # Vertical dual depolarization index, Periasamy 2018
        "VDDPI": lambda ds: (ds[co_pol] + ds[cross_pol]) / ds[co_pol],
        # cross-pol/co-pol ratio
        "ratio": ratio,
        # co-pol purity, Bhogapurapu et al. 2021
        "purity": purity,
        # pseudo scattering-type, Bhogapurapu et al. 2021
        "theta": theta,
        # pseudo scattering entropy, Bhogapurapu et al. 2021
        "entropy": entropy,
    }

    # If index supplied is not a list, convert to list. This allows us to
    # iterate through either multiple or single indices in the loop below
    indices = index if isinstance(index, list) else [index]

    # calculate for each index in the list of indices supplied (indexes)
    for index in indices:

        # Select an index function from the dictionary
        index_func = index_dict.get(str(index))

        # If no index is provided or if no function is returned due to an
        # invalid option being provided, raise an exception informing user to
        # choose from the list of valid options
        if index is None:

            raise ValueError(
                f"No radar `index` was provided. Please "
                "refer to the function \ndocumentation for a full "
                "list of valid options for `index` (e.g. 'RVI')"
            )

        elif index_func is None:

            raise ValueError(
                f"The selected index '{index}' is not one of the "
                "valid remote sensing index options. \nPlease "
                "refer to the function documentation for a full "
                "list of valid options for `index`"
            )

        # Apply index function
        index_array = index_func(ds)

        # Add as a new variable in dataset
        output_band_name = custom_varname if custom_varname else index
        ds[output_band_name] = index_array

    # Once all indexes are calculated, drop input bands if drop=True
    if drop:
        ds = ds.drop(bands_to_drop)

    # Return input dataset with added water index variable
    return ds
