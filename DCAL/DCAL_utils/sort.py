import numpy as np

def xarray_sortby_coord(dataset, coord):
    """
    Sort an xarray.Dataset by a coordinate. xarray.Dataset.sortby() sometimes fails, so this is an alternative.
    Credit to https://stackoverflow.com/a/42600594/5449970.
    """
    return dataset.loc[{coord:np.sort(dataset.coords[coord].values)}]