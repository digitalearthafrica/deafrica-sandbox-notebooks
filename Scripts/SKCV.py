"""
Functions for automating model selection through cross-validation.
"""
import warnings
warnings.simplefilter("default")
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture 
from abc import ABCMeta, abstractmethod
import xarray as xr
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

def partition_by_sum(array, parts):
    """
    Partition an array into parts of approximately equal sum.
    Does not change the order of the array elements.
    Produces the partition indices on the array. Use :func:`numpy.split` to
    divide the array along these indices.

    Parameters
    ----------
    array : array or array-like
        The 1D array that will be partitioned. The array will be raveled before
        computations.
    parts : int
        Number of parts to split the array. Can be at most the number of
        elements in the array.
    Returns
    -------
    indices : array
        The indices in which the array should be split.
    Notes
    -----
    Solution from https://stackoverflow.com/a/54024280
  
    """
    array = np.atleast_1d(array).ravel()
    if parts > array.size:
        raise ValueError(
            "Cannot partition an array of size {} into {} parts of equal sum.".format(
                array.size, parts
            )
        )
    cumulative_sum = array.cumsum()
    # Ideally, we want each part to have the same number of points (total /
    # parts).
    ideal_sum = cumulative_sum[-1] // parts
    # If the parts are ideal, the cumulative sum of each part will be this
    ideal_cumsum = np.arange(1, parts) * ideal_sum
    indices = np.searchsorted(cumulative_sum, ideal_cumsum, side="right")
    # Check for repeated split points, which indicates that there is no way to
    # split the array.
    if np.unique(indices).size != indices.size:
        raise ValueError(
            "Could not find partition points to split the array into {} parts "
            "of equal sum.".format(parts)
        )
    return indices
    
def spatial_clusters(n_groups, coordinates, method='KMeans'):
    """
    Create groups on coorindate data using either KMeans clustering
    or a Gaussian Mixture model.

    Last modified: September 2020

    Parameters
    ----------
    n_groups : int
        The number of groups to create. This is passed as 'n_clusters=n_groups'
        for the KMeans algo, and 'n_components=n_groups' for the GMM.
    coordinates : np.array
        A numpy array of coordinate values e.g. 
        np.array([[3337270.,  262400.],
                  [3441390., -273060.],
                   ...,
    method : str
        Which algorithm to use to seperate data points. Either 'KMeans' or 'GMM'

    Returns
    -------
     labels : array, shape [n_samples,]
        Index of the cluster each sample belongs to.

    """

    if method=='KMeans':
        cluster_label = KMeans(n_clusters=n_groups).fit_predict(coordinates)

    if method=='GMM':  
        cluster_label = GaussianMixture(n_components=n_groups).fit_predict(coordinates)

    return cluster_label 

class BaseSpatialCrossValidator(BaseCrossValidator, metaclass=ABCMeta):
    """
    Base class for spatial cross-validators.

    Parameters
    ----------
    n_groups : int
        The number of groups to create. This is passed as 'n_clusters=n_groups'
        for the KMeans algo, and 'n_components=n_groups' for the GMM.
    coordinates : np.array
        A numpy array of coordinate values e.g. 
        np.array([[3337270.,  262400.],
                  [3441390., -273060.], ...,             
    method : str
        Which algorithm to use to seperate data points. Either 'KMeans' or 'GMM'
    n_splits : int
        Number of splitting iterations.

    """

    def __init__(
        self,
        n_groups=None,
        coordinates=None,
        method=None,
        n_splits=None,
    ):

        self.n_groups = n_groups
        self.coordinates = coordinates
        self.method=method
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            Columns should be the easting and northing coordinates of data
            points, respectively.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems. Always
            ignored.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Always ignored.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        """
        if X.shape[1] != 2:
            raise ValueError(
                "X must have exactly 2 columns ({} given).".format(X.shape[1])
            )
        for train, test in super().split(X, y, groups):
            yield train, test
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    @abstractmethod
    def _iter_test_indices(self, X=None, y=None, groups=None):
        """
        Generates integer indices corresponding to test sets.

        MUST BE IMPLEMENTED BY DERIVED CLASSES.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            Columns should be the easting and northing coordinates of data
            points, respectively.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems. Always
            ignored.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Always ignored.

        Yields
        ------
        test : ndarray
            The testing set indices for that split.

        """

class SpatialShuffleSplit(BaseSpatialCrossValidator):
    """
    Random permutation of spatial cross-validator.

    Yields indices to split data into training and test sets. Data are first
    grouped into clusters using either a KMeans or GMM algorithm
    clusters are then split into testing and training sets randomly.

    The proportion of clusters assigned to each set is controlled by *test_size*
    and/or *train_size*. However, the total amount of actual data points in
    each set could be different from these values since clusters can have
    a different number of data points inside them. To guarantee that the
    proportion of actual data is as close as possible to the proportion of
    clusters, this cross-validator generates an extra number of splits and
    selects the one with proportion of data points in each set closer to the
    desired amount [Valavi_etal2019]_. The number of balancing splits per
    iteration is controlled by the *balancing* argument.

    This cross-validator is preferred over
    :class:`sklearn.model_selection.ShuffleSplit` for spatial data to avoid
    overestimating cross-validation scores. This can happen because of the
    inherent autocorrelation that is usually associated with this type of data
    (points that are close together are more likely to have similar values).
    See [Roberts_etal2017]_ for an overview of this topic.

    Parameters
    ----------
    n_groups : int
        The number of groups to create. This is passed as 'n_clusters=n_groups'
        for the KMeans algo, and 'n_components=n_groups' for the GMM.
    coordinates : np.array
        A numpy array of coordinate values e.g.
        np.array([[3337270.,  262400.],
                  [3441390., -273060.], ...,
    method : str
        Which algorithm to use to seperate data points. Either 'KMeans' or 'GMM'
    n_splits : int, default 10
        Number of re-shuffling & splitting iterations.
    test_size : float, int, None, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.
    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    balancing : int
        The number of splits generated per iteration to try to balance the
        amount of data in each set so that *test_size* and *train_size* are
        respected. If 1, then no extra splits are generated (essentially
        disabling the balacing). Must be >= 1.

    Returns
    --------
    generator
        containing indices to split data into training and test sets
    """        
    
    def __init__(
        self,
        n_groups=None,
        coordinates=None,
        method='KMeans',
        n_splits=None,
        test_size=0.15,
        train_size=None,
        random_state=None,
        balancing=10,
    ):
        super().__init__(n_groups=n_groups,
                         coordinates=coordinates,
                         method=method,
                         n_splits=n_splits)
        if balancing < 1:
            raise ValueError(
                "The *balancing* argument must be >= 1. To disable balancing, use 1."
            )
        self.test_size=test_size
        self.train_size=train_size
        self.random_state=random_state
        self.balancing=balancing
    
    def _iter_test_indices(self, X=None, y=None, groups=None):
        """
        Generates integer indices corresponding to test sets.

        Runs several iterations until a split is found that yields clusters with
        the right amount of data points in it.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            Columns should be the easting and northing coordinates of data
            points, respectively.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems. Always
            ignored.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Always ignored.

        Yields
        ------
        test : ndarray
            The testing set indices for that split.

        """
        labels = spatial_clusters(n_groups=self.n_groups,
                                  coordinates=self.coordinates,
                                  method=self.method)
        
        cluster_ids = np.unique(labels)
        # Generate many more splits so that we can pick and choose the ones
        # that have the right balance of training and testing data.
        shuffle = ShuffleSplit(
            n_splits=self.n_splits * self.balancing,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        ).split(cluster_ids)
        
        for _ in range(self.n_splits):
            test_sets, balance = [], []
            for _ in range(self.balancing):
                # This is a false positive in pylint which is why the warning
                # is disabled at the top of this file:
                # https://github.com/PyCQA/pylint/issues/1830
                # pylint: disable=stop-iteration-return
                train_clusters, test_clusters = next(shuffle)
                # pylint: enable=stop-iteration-return
                train_points = np.where(np.isin(labels, cluster_ids[train_clusters]))[0]
                test_points = np.where(np.isin(labels, cluster_ids[test_clusters]))[0]
                # The proportion of data points assigned to each group should
                # be close the proportion of clusters assigned to each group.
                balance.append(
                    abs(
                        train_points.size / test_points.size
                        - train_clusters.size / test_clusters.size
                    )
                )
                test_sets.append(test_points)
            best = np.argmin(balance)
            yield test_sets[best]

class SpatialKFold(BaseSpatialCrossValidator):
    """
    K-Folds over spatial cross-validator.

    Yields indices to split data into training and test sets. Data are first
    grouped into clusters using either a KMeans or GMM algorithm
    clusters. The clusters are then split into testing and training sets iteratively
    along k folds of the data (k is given by *n_splits*).

    By default, the clusters are split into folds in a way that makes each fold
    have approximately the same number of data points. Sometimes this might not
    be possible, which can happen if the number of splits is close to the
    number of clusters. In these cases, each fold will have the same number of
    clusters regardless of how many data points are in each cluster. This
    behaviour can also be disabled by setting ``balance=False``.

    This cross-validator is preferred over
    :class:`sklearn.model_selection.KFold` for spatial data to avoid
    overestimating cross-validation scores. This can happen because of the
    inherent autocorrelation that is usually associated with this type of data
    (points that are close together are more likely to have similar values).
    See [Roberts_etal2017]_ for an overview of this topic.

    Parameters
    ----------
    n_groups : int
        The number of groups to create. This is passed as 'n_clusters=n_groups'
        for the KMeans algo, and 'n_components=n_groups' for the GMM.
    coordinates : np.array
        A numpy array of coordinate values e.g. 
        np.array([[3337270.,  262400.],
                  [3441390., -273060.],
                   ...,
    n_splits : int
        Number of folds. Must be at least 2.
    shuffle : bool
        Whether to shuffle the data before splitting into batches.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    balance : bool
        Whether or not to split clusters into fold with approximately equal
        number of data points. If False, each fold will have the same number of
        clusters (which can have different number of data points in them).

    See also
    --------
    train_test_split : Split a dataset into a training and a testing set.
    cross_val_score : Score an estimator/gridder using cross-validation.

   

    """

    def __init__(
        self,
        n_groups=None,
        coordinates=None,
        method='KMeans',
        n_splits=5,
        shuffle=False,
        random_state=None,
        balance=True,
    ):
        super().__init__(n_groups=n_groups,
                         coordinates=coordinates,
                         method=method,
                         n_splits=n_splits)
        if n_splits < 2:
            raise ValueError(
                "Number of splits must be >=2 for clusterKFold. Given {}.".format(
                    n_splits
                )
            )
        self.shuffle = shuffle
        self.random_state = random_state
        self.balance = balance

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """
        Generates integer indices corresponding to test sets.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            Columns should be the easting and northing coordinates of data
            points, respectively.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems. Always
            ignored.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Always ignored.

        Yields
        ------
        test : ndarray
            The testing set indices for that split.

        """
        labels = spatial_clusters(n_groups=self.n_groups,
                                  coordinates=self.coordinates,
                                  method=self.method)
            
        cluster_ids = np.unique(labels)
        if self.n_splits > cluster_ids.size:
            raise ValueError(
                "Number of k-fold splits ({}) cannot be greater than the number of "
                "clusters ({}). Either decrease n_splits or increase the number of "
                "clusters.".format(self.n_splits, cluster_ids.size)
            )
        if self.shuffle:
            check_random_state(self.random_state).shuffle(cluster_ids)
        if self.balance:
            cluster_sizes = [np.isin(labels, i).sum() for i in cluster_ids]
            try:
                split_points = partition_by_sum(cluster_sizes, parts=self.n_splits)
                folds = np.split(np.arange(cluster_ids.size), split_points)
            except ValueError:
                warnings.warn(
                    "Could not balance folds to have approximately the same "
                    "number of data points. Dividing into folds with equal "
                    "number of clusters instead. Decreasing n_splits or increasing "
                    "the number of clusters may help.",
                    UserWarning,
                )
                folds = [i for _, i in KFold(n_splits=self.n_splits).split(cluster_ids)]
        else:
            folds = [i for _, i in KFold(n_splits=self.n_splits).split(cluster_ids)]
        for test_clusters in folds:
            test_points = np.where(np.isin(labels, cluster_ids[test_clusters]))[0]
            yield test_points

