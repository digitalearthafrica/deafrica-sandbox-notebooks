from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np
from collections import OrderedDict
import xarray as xr


def kmeans_cluster_dataset(dataset_in, bands=['red', 'green', 'blue', 'swir1', 'swir2', 'nir'], n_clusters=4):
    array_from = []
    for band in bands:
        array_from.append(dataset_in[band].values.flatten())

    np_array = np.array(array_from)
    np_array = np.swapaxes(np_array, 0, 1)

    np.set_printoptions(suppress=True)
    """
    classified = AgglomerativeClustering(n_clusters=16).fit(np_array)
    classified = DBSCAN(eps=0.005, min_samples=5, n_jobs=-1).fit(np_array)
    """
    classified = KMeans(n_clusters=n_clusters, n_jobs=-1).fit(np_array)

    classified_data = OrderedDict()

    classification = classified.labels_.reshape((dataset_in[bands[0]].shape[0], dataset_in[bands[0]].shape[1]))
    classified_data['classification'] = (['latitude', 'longitude'], classification)

    dataset_out = xr.Dataset(
        classified_data, coords={'latitude': dataset_in.latitude,
                                 'longitude': dataset_in.longitude})

    return dataset_out
