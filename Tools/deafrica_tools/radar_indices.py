"""
Functions for computing radar indices or descriptors on Digital Earth Africa
data.

.. autosummary::
   :nosignatures:
   :toctree: gen

"""

# Import required packages
import warnings
import numpy as np

# Define custom functions


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
    Takes an xarray dataset containing spectral bands, calculates one of
    a set of remote sensing indices, and adds the resulting array as a
    new variable in the original dataset.

    Last modified: June 2021

    Parameters
    ----------
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array with containing the
        spectral bands required to calculate the index. These bands are
        used as inputs to calculate the selected water index.

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
        #
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
