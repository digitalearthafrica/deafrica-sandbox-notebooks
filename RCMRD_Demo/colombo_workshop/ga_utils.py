import pandas as pd 
import numpy as np
import xarray as xr
import gdal
import affine
import fiona
import collections
from rasterio.features import shapes
from shapely import geometry
from skimage import measure
from skimage import filters
from skimage import exposure
from shapely.geometry import MultiLineString, mapping

def tasseled_cap(sensor_data, tc_bands=['greenness', 'brightness', 'wetness'],
                 drop=True):
    """   
    Computes tasseled cap wetness, greenness and brightness bands from a six
    band xarray dataset, and returns a new xarray dataset with old bands
    optionally dropped.
    
    Coefficients are from Crist and Cicone 1985 "A TM Tasseled Cap equivalent 
    transformation for reflectance factor data"
    https://doi.org/10.1016/0034-4257(85)90102-6
    
    Last modified: June 2018
    Authors: Robbi Bishop-Taylor, Bex Dunn
    
    :attr sensor_data: input xarray dataset with six Landsat bands
    :attr tc_bands: list of tasseled cap bands to compute
    (valid options: 'wetness', 'greenness','brightness')
    :attr drop: if 'drop = False', return all original Landsat bands
    :returns: xarray dataset with newly computed tasseled cap bands
    """

    # Copy input dataset
    output_array = sensor_data.copy(deep=True)

    # Coefficients for each tasseled cap band
    wetness_coeff = {'blue': 0.0315, 'green': 0.2021, 'red': 0.3102,
                     'nir': 0.1594, 'swir1': -0.6806, 'swir2': -0.6109}
    
    greenness_coeff = {'blue': -0.1603, 'green': -0.2819, 'red': -0.4934,
                       'nir': 0.7940, 'swir1': -0.0002, 'swir2': -0.1446}
    
    brightness_coeff = {'blue': 0.2043, 'green': 0.4158, 'red': 0.5524,
                        'nir': 0.5741, 'swir1': 0.3124, 'swir2': 0.2303}
    
    # Dict to use correct coefficients for each tasseled cap band
    analysis_coefficient = {'wetness': wetness_coeff,
                            'greenness': greenness_coeff,
                            'brightness': brightness_coeff}

    # For each band, compute tasseled cap band and add to output dataset
    for tc_band in tc_bands:
        # Create xarray of coefficient values used to multiply each band of input
        coeff = xr.Dataset(analysis_coefficient[tc_band])
        sensor_coeff = sensor_data * coeff

        # Sum all bands
        output_array[tc_band] = sensor_coeff.blue + sensor_coeff.green + \
                                sensor_coeff.red + sensor_coeff.nir + \
                                sensor_coeff.swir1 + sensor_coeff.swir2

    # If drop = True, remove original bands
    if drop:
        bands_to_drop = list(sensor_data.data_vars)
        output_array = output_array.drop(bands_to_drop)

    return output_array


def contour_extract(z_values, ds_array, ds_crs, ds_affine, output_shp=None, min_vertices=2,
                    attribute_data=None, attribute_dtypes=None):

    """
    Uses `skimage.measure.find_contours` to extract contour lines from a two-dimensional array.
    Contours are extracted as a dictionary of xy point arrays for each contour z-value, and optionally as
    line shapefile with one feature per contour z-value.

    The `attribute_data` and `attribute_dtypes` parameters can be used to pass custom attributes to the output
    shapefile.

    Last modified: September 2018
    Author: Robbi Bishop-Taylor

    :param z_values:
        A list of numeric contour values to extract from the array.

    :param ds_array:
        A two-dimensional array from which contours are extracted. This can be a numpy array or xarray DataArray.
        If an xarray DataArray is used, ensure that the array has one two dimensions (e.g. remove the time dimension
        using either `.isel(time=0)` or `.squeeze('time')`).

    :param ds_crs:
        Either a EPSG string giving the coordinate system of the array (e.g. 'EPSG:3577'), or a crs
        object (e.g. from an xarray dataset: `xarray_ds.geobox.crs`).

    :param ds_affine:
        Either an affine object from a rasterio or xarray object (e.g. `xarray_ds.geobox.affine`), or a gdal-derived
        geotransform object (e.g. `gdal_ds.GetGeoTransform()`) which will be converted to an affine.

    :param min_vertices:
        An optional integer giving the minimum number of vertices required for a contour to be extracted. The default
        (and minimum) value is 2, which is the smallest number required to produce a contour line (i.e. a start and
        end point). Higher values remove smaller contours, potentially removing noise from the output dataset.

    :param output_shp:
        An optional string giving a path and filename for the output shapefile. Defaults to None, which
        does not generate a shapefile.

    :param attribute_data:
        An optional dictionary of lists used to define attributes/fields to add to the shapefile. Dict keys give
        the name of the shapefile attribute field, while dict values must be lists of the same length as `z_values`.
        For example, if `z_values=[0, 10, 20]`, then `attribute_data={'type: [1, 2, 3]}` can be used to create a
        shapefile field called 'type' with a value for each contour in the shapefile. The default is None, which
        produces a default shapefile field called 'z_value' with values taken directly from the `z_values` parameter
        and formatted as a 'float:9.2'.

    :param attribute_dtypes:
        An optional dictionary giving the output dtype for each shapefile attribute field that is specified by
        `attribute_data`. For example, `attribute_dtypes={'type: 'int'}` can be used to set the 'type' field to an
        integer dtype. The dictionary should have the same keys/field names as declared in `attribute_data`.
        Valid values include 'int', 'str', 'datetime, and 'float:X.Y', where X is the minimum number of characters
        before the decimal place, and Y is the number of characters after the decimal place.

    :return:
        A dictionary with contour z-values as the dict key, and a list of xy point arrays as dict values.

    :example:

    >>> # Import modules
    >>> import sys
    >>> import datacube

    >>> # Import external dea-notebooks functions using relative link to Scripts directory
    >>> sys.path.append('../10_Scripts')
    >>> import SpatialTools

    >>> # Set up datacube instance
    >>> dc = datacube.Datacube(app='Contour extraction')

    >>> # Define an elevation query
    >>> elevation_query = {'lat': (-35.25, -35.35),
    ...                    'lon': (149.05, 149.17),
    ...                    'output_crs': 'EPSG:3577',
    ...                    'resolution': (-25, 25)}

    >>> # Import sample elevation data
    >>> elevation_data = dc.load(product='srtm_dem1sv1_0', **elevation_query)

    >>> # Remove the time dimension so that array is two-dimensional
    >>> elevation_2d = elevation_data.dem_h.squeeze('time')

    >>> # Extract contours
    >>> contour_dict = SpatialTools.contour_extract(z_values=[600, 700, 800],
    ...                                             ds_array=elevation_2d,
    ...                                             ds_crs=elevation_2d.geobox.crs,
    ...                                             ds_affine=elevation_2d.geobox.affine,
    ...                                             output_shp='extracted_contours.shp')
    Extracting contour 600
    Extracting contour 700
    Extracting contour 800
    <BLANKLINE>
    Exporting contour shapefile to extracted_contours.shp

    """

    # First test that input array has only two dimensions:
    if len(ds_array.shape) == 2:

        # Obtain affine object from either rasterio/xarray affine or a gdal geotransform:
        if type(ds_affine) != affine.Affine:

            ds_affine = affine.Affine.from_gdal(*ds_affine)

        ####################
        # Extract contours #
        ####################

        # Output dict to hold contours for each offset
        contours_dict = collections.OrderedDict()

        for z_value in z_values:

            # Extract contours and convert output array pixel coordinates into arrays of real world coordinates.
            # We need to add (0.5 x the pixel size) to x values and subtract (-0.5 * pixel size) from y values to
            # correct coordinates to give the centre point of pixels, rather than the top-left corner
            ps = ds_affine[0]  # Compute pixel size
            contours_geo = [np.column_stack(ds_affine * (i[:, 1], i[:, 0])) + 
                            np.array([0.5 * ps, -0.5 * ps]) for i in
                            measure.find_contours(ds_array, z_value)]

            # For each array of coordinates, drop any xy points that have NA
            contours_nona = [i[~np.isnan(i).any(axis=1)] for i in contours_geo]

            # Drop 0 length and add list of contour arrays to dict
            contours_withdata = [i for i in contours_nona if len(i) >= min_vertices]

            # If there is data for the contour, add to dict:
            if len(contours_withdata) > 0:
                contours_dict[z_value] = contours_withdata
            else:
                print('    No data for contour {}; skipping'.format(z_value))

        #######################
        # Export to shapefile #
        #######################

        # If a shapefile path is given, generate shapefile
        if output_shp:

            print('\nExporting contour shapefile to {}'.format(output_shp))

            # If attribute fields are left empty, default to including a single z-value field based on `z_values`
            if not attribute_data:

                # Default field uses two decimal points by default
                attribute_data = {'z_value': z_values}
                attribute_dtypes = {'z_value': 'float:9.2'}

            # Set up output multiline shapefile properties
            schema = {'geometry': 'MultiLineString',
                      'properties': attribute_dtypes}
            
            # Create output shapefile for writing
            with fiona.open(output_shp, 'w',
                            crs={'init': str(ds_crs), 'no_defs': True},
                            driver='ESRI Shapefile',
                            schema=schema) as output:

                # Write each shapefile to the dataset one by one
                for i, (z_value, contours) in enumerate(contours_dict.items()):

                    # Create multi-string object from all contour coordinates
                    contour_multilinestring = MultiLineString(contours)

                    # Get attribute values for writing
                    attribute_vals = {field_name: field_vals[i] for field_name, field_vals in attribute_data.items()}

                    # Write output shapefile to file with z-value field
                    output.write({'properties': attribute_vals,
                                  'geometry': mapping(contour_multilinestring)})

        # Return dict of contour arrays
        return contours_dict

    else:
        print('None')

        
# Extract vertex coordinates and heights from geopandas
def contours_to_arrays(gdf, col):
    
    coords_zvals = []
    
    for i in range(1, len(gdf)):
        
        val = gdf.iloc[i][col]
    
        try:
            coords = np.concatenate([np.vstack(x.coords.xy).T for x in gdf.iloc[i].geometry])
            
        except:
            coords = np.vstack(gdf.iloc[i].geometry.coords.xy).T

        coords_zvals.append(np.column_stack((coords, np.full(np.shape(coords)[0], fill_value=val))))
    
    return np.concatenate(coords_zvals)


def interpolate_timeseries(ds, freq='7D', method='linear'):
    
    """
    Interpolate new data between each existing xarray timestep at a given
    frequency. For example, `freq='7D'` will interpolate new values at weekly
    intervals from the start time of the xarray dataset to the end time. 
    `freq='24H'` will interpolate new values for each day, etc.
    
    :param ds:
        The xarray dataset to interpolate new time-step observations for.
        
    :param freq:
        An optional string giving the frequency at which to interpolate new time-step 
        observations. Defaults to '7D' which interpolates new values at weekly intervals; 
        for a full list of options refer to Panda's list of offset aliases: 
        https://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
        
    :param method:
        An optional string giving the interpolation method to use to generate new time-steps.
        Default is 'linear'; options are {'linear', 'nearest'} for multidimensional arrays and
        {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'} for 1-dimensional arrays.
        
    :return:
        A matching xarray dataset covering the same time period as `ds`, but with an 
        interpolated for each time-step given by `freq`.
        
    """    
    
    # Use pandas to generate dates from start to end of ds at a given frequency
    start_time = ds.isel(time=0).time.values.item() 
    end_time = ds.isel(time=-1).time.values.item()    
    from_to = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Use these dates to linearly interpolate new data for each new date
    print('Interpolating {} time-steps at {} intervals'.format(len(from_to), freq))
    return ds.interp(coords={'time': from_to}, method=method)
