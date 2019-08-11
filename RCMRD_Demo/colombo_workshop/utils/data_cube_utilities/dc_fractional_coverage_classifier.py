import numpy as np
import xarray as xr
import scipy.optimize as opt  #nnls

import datacube
from . import dc_utilities as utilities
from .dc_utilities import create_default_clean_mask

# Command line tool imports
import argparse
import os
import collections
import gdal
from datetime import datetime

# Author: KMF
# Creation date: 2016-10-24


def frac_coverage_classify(dataset_in, clean_mask=None, no_data=-9999):
    """
    Description:
      Performs fractional coverage algorithm on given dataset.
    Assumption:
      - The implemented algorithm is defined for Landsat 5/Landsat 7; in order for it to
        be used for Landsat 8, the bands will need to be adjusted
    References:
      - Guerschman, Juan P., et al. "Assessing the effects of site heterogeneity and soil
        properties when unmixing photosynthetic vegetation, non-photosynthetic vegetation
        and bare soil fractions from Landsat and MODIS data." Remote Sensing of Environment
        161 (2015): 12-26.
    -----
    Inputs:
      dataset_in (xarray.Dataset) - dataset retrieved from the Data Cube (can be a derived
        product, such as a cloudfree mosaic; should contain
          coordinates: latitude, longitude
          variables: blue, green, red, nir, swir1, swir2
    Optional Inputs:
      clean_mask (nd numpy array with dtype boolean) - true for values user considers clean;
        if user does not provide a clean mask, all values will be considered clean
      no_data (int/float) - no data pixel value; default: -9999
    Output:
      dataset_out (xarray.Dataset) - fractional coverage results with no data = -9999; containing
          coordinates: latitude, longitude
          variables: bs, pv, npv
        where bs -> bare soil, pv -> photosynthetic vegetation, npv -> non-photosynthetic vegetation
    """
    # Default to masking nothing.
    if clean_mask is None:
        clean_mask = create_default_clean_mask(dataset_in)
    
    band_stack = []

    mosaic_clean_mask = clean_mask.flatten()

    for band in [
            dataset_in.blue.values, dataset_in.green.values, dataset_in.red.values, dataset_in.nir.values,
            dataset_in.swir1.values, dataset_in.swir2.values
    ]:
        band = band.astype(np.float32)
        band = band * 0.0001
        band = band.flatten()
        band_clean = np.full(band.shape, np.nan)
        band_clean[mosaic_clean_mask] = band[mosaic_clean_mask]
        band_stack.append(band_clean)

    band_stack = np.array(band_stack).transpose()

    for b in range(6):
        band_stack = np.hstack((band_stack, np.expand_dims(np.log(band_stack[:, b]), axis=1)))
    for b in range(6):
        band_stack = np.hstack(
            (band_stack, np.expand_dims(np.multiply(band_stack[:, b], band_stack[:, b + 6]), axis=1)))
    for b in range(6):
        for b2 in range(b + 1, 6):
            band_stack = np.hstack(
                (band_stack, np.expand_dims(np.multiply(band_stack[:, b], band_stack[:, b2]), axis=1)))
    for b in range(6):
        for b2 in range(b + 1, 6):
            band_stack = np.hstack(
                (band_stack, np.expand_dims(np.multiply(band_stack[:, b + 6], band_stack[:, b2 + 6]), axis=1)))
    for b in range(6):
        for b2 in range(b + 1, 6):
            band_stack = np.hstack((band_stack, np.expand_dims(
                np.divide(band_stack[:, b2] - band_stack[:, b], band_stack[:, b2] + band_stack[:, b]), axis=1)))

    band_stack = np.nan_to_num(band_stack)  # Now a n x 63 matrix (assuming one acquisition)

    ones = np.ones(band_stack.shape[0])
    ones = ones.reshape(ones.shape[0], 1)
    band_stack = np.concatenate((band_stack, ones), axis=1)  # Now a n x 64 matrix (assuming one acquisition)

    end_members = np.loadtxt(
        './endmembers_landsat.csv',
        delimiter=',')  # Creates a 64 x 3 matrix

    SumToOneWeight = 0.02
    ones = np.ones(end_members.shape[1]) * SumToOneWeight
    ones = ones.reshape(1, end_members.shape[1])
    end_members = np.concatenate((end_members, ones), axis=0).astype(np.float32)

    result = np.zeros((band_stack.shape[0], end_members.shape[1]), dtype=np.float32)  # Creates an n x 3 matrix

    for i in range(band_stack.shape[0]):
        if mosaic_clean_mask[i]:
            result[i, :] = (opt.nnls(end_members, band_stack[i, :])[0].clip(0, 2.54) * 100).astype(np.int16)
        else:
            result[i, :] = np.ones((end_members.shape[1]), dtype=np.int16) * (-9999)  # Set as no data

    latitude = dataset_in.latitude
    longitude = dataset_in.longitude

    result = result.reshape(latitude.size, longitude.size, 3)

    pv_band = result[:, :, 0]
    npv_band = result[:, :, 1]
    bs_band = result[:, :, 2]

    pv_clean = np.full(pv_band.shape, -9999)
    npv_clean = np.full(npv_band.shape, -9999)
    bs_clean = np.full(bs_band.shape, -9999)
    pv_clean[clean_mask] = pv_band[clean_mask]
    npv_clean[clean_mask] = npv_band[clean_mask]
    bs_clean[clean_mask] = bs_band[clean_mask]

    rapp_bands = collections.OrderedDict([('bs', (['latitude', 'longitude'], bs_band)),
                                          ('pv', (['latitude', 'longitude'], pv_band)),
                                          ('npv', (['latitude', 'longitude'], npv_band))])

    rapp_dataset = xr.Dataset(rapp_bands, coords={'latitude': latitude, 'longitude': longitude})

    return rapp_dataset


def main(platform, product_type, min_lon, max_lon, min_lat, max_lat, start_date, end_date, dc_config):
    """
    Description:
      Command-line fractional coverage tool - TODO
    Assumptions:
      The command-line tool assumes there is a measurement called cf_mask
    Inputs:
      platform (str)
      product_type (str)
      min_lon (str)
      max_lon (str)
      min_lat (str)
      max_lat (str)
      start_date (str)
      end_date (str)
      dc_config (str)
    """

    # Initialize data cube object
    dc = datacube.Datacube(config=dc_config, app='dc-frac-cov')

    products = dc.list_products()
    platform_names = set([product[6] for product in products.values])
    if platform not in platform_names:
        print('ERROR: Invalid platform.')
        print('Valid platforms are:')
        for name in platform_names:
            print(name)
        return

    product_names = [product[0] for product in products.values]
    if product_type not in product_names:
        print('ERROR: Invalid product type.')
        print('Valid product types are:')
        for name in product_names:
            print(name)
        return

    try:
        min_lon = float(args.min_lon)
        max_lon = float(args.max_lon)
        min_lat = float(args.min_lat)
        max_lat = float(args.max_lat)
    except:
        print('ERROR: Longitudes/Latitudes must be float values')
        return

    try:
        start_date_str = start_date
        end_date_str = end_date
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    except:
        print('ERROR: Invalid date format. Date format: YYYY-MM-DD')
        return

    if not os.path.exists(dc_config):
        print('ERROR: Invalid file path for dc_config')
        return

    # Retrieve data from Data Cube
    dataset_in = dc.load(
        platform=platform,
        product=product_type,
        time=(start_date, end_date),
        lon=(min_lon, max_lon),
        lat=(min_lat, max_lat))

    # Get information needed for saving as GeoTIFF

    # Spatial ref
    crs = dataset_in.crs
    spatial_ref = utilities.get_spatial_ref(crs)

    # Upper left coordinates
    ul_lon = dataset_in.longitude.values[0]
    ul_lat = dataset_in.latitude.values[0]

    # Resolution
    products = dc.list_products()
    resolution = products.resolution[products.name == 'ls7_ledaps']
    lon_dist = resolution.values[0][1]
    lat_dist = resolution.values[0][0]

    # Rotation
    lon_rtn = 0
    lat_rtn = 0

    geotransform = (ul_lon, lon_dist, lon_rtn, ul_lat, lat_rtn, lat_dist)

    dataset_out = frac_coverage_classify(dataset_in)

    out_file = (str(min_lon) + '_' + str(min_lat) + '_' + start_date_str + '_' + end_date_str + '_frac_coverage.tif')

    utilities.save_to_geotiff(out_file, gdal.GDT_Float32, dataset_out, geotransform, spatial_ref)


if __name__ == '__main__':

    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('platform', help='Data platform; example: LANDSAT_7')
    parser.add_argument('product', help='Product type; example: ls7_ledaps')
    parser.add_argument('min_lon', help='Minimum longitude')
    parser.add_argument('max_lon', help='Maximum longitude')
    parser.add_argument('min_lat', help='Minimum latitude')
    parser.add_argument('max_lat', help='Maximum latitude')
    parser.add_argument('start_date', help='Start date; format: YYYY-MM-DD')
    parser.add_argument('end_date', help='End date; format: YYYY-MM-DD')
    parser.add_argument(
        'dc_config',
        nargs='?',
        default='~/.datacube.conf',
        help='Datacube configuration path; default: ~/.datacube.conf')

    args = parser.parse_args()

    main(args.platform, args.product, args.min_lon, args.max_lon, args.min_lat, args.max_lat, args.start_date,
         args.end_date, args.dc_config)

    end_time = datetime.now()
    print('Execution time: ' + str(end_time - start_time))
