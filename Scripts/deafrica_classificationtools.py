# deafrica_classificationtools.py
'''
Description: This file contains a set of python functions for applying 
machine learning classifiying remote sensing data from Digital Earth 
Africa

License: The code in this notebook is licensed under the Apache License, 
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Digital Earth 
Africa data is licensed under the Creative Commons by Attribution 4.0 
license (https://creativecommons.org/licenses/by/4.0/).

Contact: If you need assistance, please post a question on the Open Data 
Cube Slack channel (http://slack.opendatacube.org/) or on the GIS Stack 
Exchange (https://gis.stackexchange.com/questions/ask?tags=open-data-cube) 
using the `open-data-cube` tag (you can view previously asked questions 
here: https://gis.stackexchange.com/questions/tagged/open-data-cube).

If you would like to report an issue with this script, you can file one on 
Github https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues

Last modified: Feb 2020


'''


import numpy as np
import xarray as xr
import geopandas as gpd
from copy import deepcopy
import datacube
from dask.diagnostics import ProgressBar
from rasterio.features import geometry_mask
from rasterio.features import rasterize
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin
from datacube.storage.masking import make_mask
from datacube.storage import masking
from datacube.utils import geometry
from datacube_stats.statistics import GeoMedian
import rasterio
import sys
import os

sys.path.append('../Scripts')
from deafrica_datahandling import mostcommon_crs, load_ard
from deafrica_bandindices import calculate_indices


def sklearn_flatten(input_xr):
    """
    Reshape a DataArray or Dataset with spatial (and optionally 
    temporal) structure into an np.array with the spatial and temporal 
    dimensions flattened into one dimension.

    This flattening procedure enables DataArrays and Datasets to be used 
    to train and predict
    with sklearn models.

    Last modified: September 2019

    Parameters
    ----------
    input_xr : xarray.DataArray or xarray.Dataset 
        Must have dimensions 'x' and 'y', may have dimension 'time'.
        Dimensions other than 'x', 'y' and 'time' are unaffected by the 
        flattening.

    Returns
    ----------
    input_np : numpy.array 
        A numpy array corresponding to input_xr.data (or 
        input_xr.to_array().data), with dimensions 'x','y' and 'time' 
        flattened into a single dimension, which is the first axis of 
        the returned array. input_np contains no NaNs.

    """
    # cast input Datasets to DataArray
    if isinstance(input_xr, xr.Dataset):
        input_xr = input_xr.to_array()

    # stack across pixel dimensions, handling timeseries if necessary
    if 'time' in input_xr.dims:
        stacked = input_xr.stack(z=['x', 'y', 'time'])
    else:
        stacked = input_xr.stack(z=['x', 'y'])

    # finding 'bands' dimensions in each pixel - these will not be
    # flattened as their context is important for sklearn
    pxdims = []
    for dim in stacked.dims:
        if dim != 'z':
            pxdims.append(dim)

    # mask NaNs - we mask pixels with NaNs in *any* band, because
    # sklearn cannot accept NaNs as input
    mask = np.isnan(stacked)
    if len(pxdims) != 0:
        mask = mask.any(dim=pxdims)

    # turn the mask into a numpy array (boolean indexing with xarrays
    # acts weird)
    mask = mask.data

    # the dimension we are masking along ('z') needs to be the first
    # dimension in the underlying np array for the boolean indexing to work
    stacked = stacked.transpose('z', *pxdims)
    input_np = stacked.data[~mask]

    return input_np


def sklearn_unflatten(output_np, input_xr):
    """
    Reshape a numpy array with no 'missing' elements (NaNs) and 
    'flattened' spatiotemporal structure into a DataArray matching the 
    spatiotemporal structure of the DataArray

    This enables an sklearn model's prediction to be remapped to the 
    correct pixels in the input DataArray or Dataset.

    Last modified: September 2019

    Parameters
    ----------
    output_np : numpy.array 
        The first dimension's length should correspond to the number of 
        valid (non-NaN) pixels in input_xr.
    input_xr : xarray.DataArray or xarray.Dataset 
        Must have dimensions 'x' and 'y', may have dimension 'time'. 
        Dimensions other than 'x', 'y' and 'time' are unaffected by the 
        flattening.

    Returns
    ----------
    output_xr : xarray.DataArray 
        An xarray.DataArray with the same dimensions 'x', 'y' and 'time'
        as input_xr, and the same valid (non-NaN) pixels. These pixels 
        are set to match the data in output_np.

    """

    # the output of a sklearn model prediction should just be a numpy array
    # with size matching x*y*time for the input DataArray/Dataset.

    # cast input Datasets to DataArray
    if isinstance(input_xr, xr.Dataset):
        input_xr = input_xr.to_array()

    # generate the same mask we used to create the input to the sklearn model
    if 'time' in input_xr.dims:
        stacked = input_xr.stack(z=['x', 'y', 'time'])
    else:
        stacked = input_xr.stack(z=['x', 'y'])

    pxdims = []
    for dim in stacked.dims:
        if dim != 'z':
            pxdims.append(dim)

    mask = np.isnan(stacked)
    if len(pxdims) != 0:
        mask = mask.any(dim=pxdims)

    # handle multivariable output
    output_px_shape = ()
    if len(output_np.shape[1:]):
        output_px_shape = output_np.shape[1:]

    # use the mask to put the data in all the right places
    output_ma = np.ma.empty((len(stacked.z), *output_px_shape))
    output_ma[~mask] = output_np
    output_ma.mask = mask

    # set the stacked coordinate to match the input
    output_xr = xr.DataArray(output_ma, coords={'z': stacked['z']},
                             dims=['z', *['output_dim_' + str(idx) for
                                          idx in range(len(output_px_shape))]])

    output_xr = output_xr.unstack()

    return output_xr


def fit_xr(model, input_xr):
    """
    Utilise our wrappers to fit a vanilla sklearn model.

    Last modified: September 2019

    Parameters
    ----------
    model : scikit-learn model or compatible object
        Must have a fit() method that takes numpy arrays.
    input_xr : xarray.DataArray or xarray.Dataset. 
        Must have dimensions 'x' and 'y', may have dimension 'time'.

    Returns
    ----------
    model : a scikit-learn model which has been fitted to the data in 
    the pixels of input_xr.

    """

    model = model.fit(sklearn_flatten(input_xr))
    return model


def predict_xr(model, input_xr, progress=True):
    """
    Utilise our wrappers to predict with a vanilla sklearn model.

    Last modified: September 2019

    Parameters
    ----------
    model : a scikit-learn model or compatible object
        Must have a predict() method that takes numpy arrays.
    input_xr : xarray.DataArray or xarray.Dataset
        Must have dimensions 'x' and 'y', may have dimension 'time'.

    Returns
    ----------
    output_xr : xarray.DataArray 
        An xarray.DataArray containing the prediction output from model 
        with input_xr as input. Has the same spatiotemporal structure 
        as input_xr.

    """

    def _get_class_ufunc(*args):
        """
        ufunc to apply classification to chunks of data
        """
        input_data_flattened = []
        for data in args:
            input_data_flattened.append(data.flatten())

        # Flatten array
        input_data_flattened = np.array(input_data_flattened).transpose()

        # Mask out no-data in input (not all classifiers can cope with
        # Inf or NaN values)
        input_data_flattened = np.where(np.isfinite(input_data_flattened),
                                        input_data_flattened, 0)

        # Actually apply the classification
        out_class = model.predict(input_data_flattened)

        # Mask out NaN or Inf values in results
        out_class = np.where(np.isfinite(out_class), out_class, 0)

        # Reshape when writing out
        return out_class.reshape(args[0].shape)

    def _get_class(*args):
        """
        Apply classification to xarray DataArrays.

        Uses dask to run chunks at a time in parallel

        """
        out = xr.apply_ufunc(_get_class_ufunc, *args,
                             dask='parallelized', output_dtypes=[np.uint8])

        return out

    # Set up a list of input data using variables passed in
    input_data = []

    for var_name in input_xr.data_vars:
        input_data.append(input_xr[var_name])

    # Run through classification. Need to expand and have a separate
    # dataframe for each variable so chunking in dask works.
    if progress:
        with ProgressBar():
            out_class = _get_class(*input_data).compute()
    else:
        out_class = _get_class(*input_data).compute()

    # Set the stacked coordinate to match the input
    output_xr = xr.DataArray(out_class, coords=input_xr.coords)

    return output_xr


class HiddenPrints:
    """
    For concealing unwanted print statements called by other functions
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_training_data_for_shp(polygons,
                              out,
                              products,
                              dc_query,
                              field=None,
                              calc_indices=None,
                              reduce_func=None,
                              drop=True,
                              zonal_stats=None,
                              collection=None):
    """
    Function to extract data for training a classifier using a shapefile 
    of labelled polygons.

    Parameters
    ----------
    polygons : geopandas geodataframe
        polygon data in the form of a geopandas geodataframe
    out : list
        Empty list to contain output data.
    products : list
        a list of products ot load from the datacube. 
        e.g. ['ls8_usgs_sr_scene', 'ls7_usgs_sr_scene']
    dc_query : dictionary
        Datacube query object, should not contain lat and long (x or y)
        variables as these are supplied by the 'polygons' variable
    field : string 
        A string containing name of column with labels in shapefile 
        attribute table. Field must contain numeric values.
    calc_indices: list, optional
        An optional list giving the names of any remote sensing indices 
        to be calculated on the loaded data (e.g. `['NDWI', 'NDVI']`. 
    reduce_func : string, optional 
        Function to reduce the data from multiple time steps to
        a single timestep. Options are 'mean'
    drop : booleam, optional , 'median', or 'geomedian'
        If this variable is set to True, and 'calc_indices' are supplied, the
        spectral bands will be dropped from the dataset leaving only the
        band indices as data variables in the dataset. Default is False.
    zonal_stats: string, optional
        An optional string giving the names of zonal statistics to calculate 
        for the polygon. Default is None (all pixel values). Supported 
        values are 'mean' or 'median' 
    collection: string, optional
        to calculate band indices, the satellite collection is required.
        Options include 'c1' for Landsat C1, 'c2' for Landsat C2, and 
        's2' for Sentinel 2.

    Returns
    --------
    A list of numpy.arrays containing classes and extracted data for 
    each pixel or polygon.

    """
    #prevent function altering dictionary kwargs
    dc_query = deepcopy(dc_query)
    dc = datacube.Datacube(app='training_data')

    #set up some print statements
    i = 0
    if calc_indices is not None:
            print("Calculating indices: " + str(calc_indices))
    if reduce_func is not None:
            print("Reducing data using: " + reduce_func)
    if zonal_stats is not None:
            print("Taking zonal statistic: "+ zonal_stats)
    
    # loop through polys and extract training data
    for index, row in polygons.iterrows():
        print(" Feature {:04}/{:04}\r".format(i + 1, len(polygons)), 
              end='')

        # set up query based on polygon (convert to WGS84)
        geom = geometry.Geometry(
            polygons.geometry.values[index].__geo_interface__, geometry.CRS(
                'epsg:4326'))
        #print(geom)    
        q = {"geopolygon": geom}

        # merge polygon query with user supplied query params
        dc_query.update(q)
        
        # Identify the most common projection system in the input query
        output_crs = mostcommon_crs(dc=dc, product=products, query=dc_query)
        
        #load_ard doesn't handle geomedians
        if 'ga_ls8c_gm_2_annual' in products:
            ds = dc.load(product='ga_ls8c_gm_2_annual', **dc_query)
            ds = ds.where(ds!=0, np.nan)
            
        else:
            # load data
            with HiddenPrints():
                ds = load_ard(dc=dc,
                              products=products,
                              output_crs=output_crs,
                              **dc_query)

        # create polygon mask
        mask = rasterio.features.geometry_mask(
            [geom.to_crs(ds.geobox.crs) for geoms in [geom]],
            out_shape=ds.geobox.shape,
            transform=ds.geobox.affine,
            all_touched=False,
            invert=False)

        mask = xr.DataArray(mask, dims=("y", "x"))
        ds = ds.where(mask == False)

        # Check if band indices are wanted
        if calc_indices is not None:

            if len(ds.time.values) > 1:
                
                if reduce_func in ['mean', 'median', 'std']:
                    with HiddenPrints():
                        data = calculate_indices(ds,
                                                 index=calc_indices,
                                                 drop=drop,
                                                 collection=collection)
                        method_to_call = getattr(data, reduce_func)
                        data = method_to_call(dim='time')

                elif reduce_func == 'geomedian':
                    data = GeoMedian().compute(ds)
                    with HiddenPrints():
                        data = calculate_indices(data,
                                                 index=calc_indices,
                                                 drop=drop,
                                                 collection=collection)
                
                else:
                    raise Exception(f'{zonal_stats} is not one of the supported reduce functions ("std", "mean", "median")')

            else:
                with HiddenPrints():
                    data = calculate_indices(ds,
                                             index=calc_indices,
                                             drop=drop,
                                             collection=collection)

        # when band indices are not required, reduce the
        # dataset to a 2d array through means or (geo)medians
        if calc_indices is None:
            if (len(ds.time.values) > 1) and (reduce_func==None):
                raise ValueError("You're dataset has "+ str(len(ds.time.values)) + 
                                 "time-steps, please provide a reduction function, e.g. reduce_func='mean'")
                
            if len(ds.time.values) > 1:
                
                if reduce_func == 'geomedian':
                    data = GeoMedian().compute(ds)
                
                elif reduce_func in ['mean', 'median', 'std']:
                    method_to_call = getattr(ds, reduce_func)
                    data = method_to_call('time')
            else:
                data = ds.squeeze()

        # compute in case we have dask arrays
        data = data.compute()
        
        if zonal_stats is None:
            # If no summary stats were requested then extract all pixel values
            flat_train = sklearn_flatten(data)
            # Make a labelled array of identical size
            flat_val = np.repeat(row[field], flat_train.shape[0])
            stacked = np.hstack((np.expand_dims(flat_val, axis=1), flat_train))
        
        elif zonal_stats in ['mean', 'median', 'std']:
            method_to_call = getattr(data, zonal_stats)
            flat_train = method_to_call()
            flat_train = flat_train.to_array()
            stacked = np.hstack((row[field], flat_train))
        
        else:
            raise Exception(f'{zonal_stats} is not one of the supported reduce functions ("std", "mean", "median")')
       
        # Append training data and label to list
        out.append(stacked)
        i+=1
    # Return a list of labels for columns in output array
    
    return [field] + list(data.data_vars)

class KMeans_tree(ClusterMixin):
    """
    A hierarchical KMeans unsupervised clustering model. This class is 
    a clustering model, so it inherits scikit-learn's ClusterMixin 
    base class.

    Parameters
    ----------
    n_levels : integer, default 2
        number of levels in the tree of clustering models.
    n_clusters : integer, default 3
        Number of clusters in each of the constituent KMeans models in 
        the tree.
    **kwargs : optional
        Other keyword arguments to be passed directly to the KMeans 
        initialiser.

    """

    def __init__(self, n_levels=2, n_clusters=3, **kwargs):

        assert (n_levels >= 1)

        self.base_model = KMeans(n_clusters=3, **kwargs)
        self.n_levels = n_levels
        self.n_clusters = n_clusters
        # make child models
        if n_levels > 1:
            self.branches = [KMeans_tree(n_levels=n_levels - 1,
                                         n_clusters=n_clusters,
                                         **kwargs)
                             for _ in range(n_clusters)]

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit the tree of KMeans models. All parameters mimic those 
        of KMeans.fit().

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the 
            data will be converted to C ordering, which will cause a 
            memory copy if the given data is not C-contiguous.
        y : Ignored
            not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all 
            observations are assigned equal weight (default: None)
        """

        self.labels_ = self.base_model.fit(X,
                                           sample_weight=sample_weight).labels_

        if self.n_levels > 1:
            labels_old = np.copy(self.labels_)
            # make room to add the sub-cluster labels
            self.labels_ *= (self.n_clusters) ** (self.n_levels - 1)

            for clu in range(self.n_clusters):
                # fit child models on their corresponding partition of the training set
                self.branches[clu].fit(X[labels_old == clu], sample_weight=(
                    sample_weight[labels_old == clu]
                    if sample_weight is not None else None))
                self.labels_[labels_old == clu] += self.branches[clu].labels_

        return self

    def predict(self, X, sample_weight=None):
        """
        Send X through the KMeans tree and predict the resultant 
        cluster. Compatible with KMeans.predict().

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all 
            observations are assigned equal weight (default: None)

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """

        result = self.base_model.predict(X, sample_weight=sample_weight)

        if self.n_levels > 1:
            rescpy = np.copy(result)

            # make room to add the sub-cluster labels
            result *= (self.n_clusters) ** (self.n_levels - 1)

            for clu in range(self.n_clusters):
                result[rescpy == clu] += self.branches[clu].predict(X[rescpy == clu], sample_weight=(
                    sample_weight[rescpy == clu] if sample_weight is not None else None))

        return result
