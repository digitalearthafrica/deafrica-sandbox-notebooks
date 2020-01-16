# Copyright 2016 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. All Rights Reserved.
#
# Portion of this code is Copyright Geoscience Australia, Licensed under the
# Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License
# at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# The CEOS 2 platform is licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

# datacube imports.
import datacube
from datacube.api import GridWorkflow
import xarray as xr
import numpy as np
from datetime import date


class DataAccessApi:
    """
    Class that provides wrapper functionality for the DataCube.
    """

    def __init__(self, config=None):
        self.dc = datacube.Datacube(config=config)

    def close(self):
        self.dc.close()

    """
    query params are defined in datacube.api.query
    """

    def get_dataset_by_extent(self,
                              product,
                              product_type=None,
                              platform=None,
                              time=None,
                              longitude=None,
                              latitude=None,
                              measurements=None,
                              output_crs=None,
                              resolution=None,
                              dask_chunks=None,
                              **kwargs):
        """
        Gets and returns data based on lat/long bounding box inputs.
        All params are optional. Leaving one out will just query the dc without it, (eg leaving out
        lat/lng but giving product returns dataset containing entire product.)

        Args:
            product (string): The name of the product associated with the desired dataset.
            product_type (string): The type of product associated with the desired dataset.
            platform (string): The platform associated with the desired dataset.
            time (tuple): A tuple consisting of the start time and end time for the dataset.
            longitude (tuple): A tuple of floats specifying the min,max longitude bounds.
            latitude (tuple): A tuple of floats specifying the min,max latitutde bounds.
            crs (string): CRS lat/lon bounds are specified in, defaults to WGS84.
            output_crs (string): Determines reprojection of the data before its returned
            resolution (tuple): A tuple of min,max ints to determine the resolution of the data.
            dask_chunks (dict): Lazy loaded array block sizes, not lazy loaded by default.

        Returns:
            data (xarray): dataset with the desired data.
        """

        # there is probably a better way to do this but I'm not aware of it.
        query = {}
        if product_type is not None:
            query['product_type'] = product_type
        if platform is not None:
            query['platform'] = platform
        if time is not None:
            query['time'] = time
        if longitude is not None and latitude is not None:
            query['longitude'] = longitude
            query['latitude'] = latitude

        data = self.dc.load(
            product=product,
            measurements=measurements,
            output_crs=output_crs,
            resolution=resolution,
            dask_chunks=dask_chunks,
            **query)
        return data

    def get_stacked_datasets_by_extent(self,
                                       products,
                                       product_type=None,
                                       platforms=None,
                                       time=None,
                                       longitude=None,
                                       latitude=None,
                                       measurements=None,
                                       output_crs=None,
                                       resolution=None,
                                       dask_chunks=None,
                                       **kwargs):
        """
        Gets and returns data based on lat/long bounding box inputs.
        All params are optional. Leaving one out will just query the dc without it, (eg leaving out
        lat/lng but giving product returns dataset containing entire product.)

        Args:
          products (array of strings): The names of the product associated with the desired dataset.
          product_type (string): The type of product associated with the desired dataset.
          platforms (array of strings): The platforms associated with the desired dataset.
          time (tuple): A tuple consisting of the start time and end time for the dataset.
          longitude (tuple): A tuple of floats specifying the min,max longitude bounds.
          latitude (tuple): A tuple of floats specifying the min,max latitutde bounds.
          measurements (list): A list of strings that represents all measurements.
          output_crs (string): Determines reprojection of the data before its returned
          resolution (tuple): A tuple of min,max ints to determine the resolution of the data.

        Returns:
          data (xarray): dataset with the desired data.
        """

        data_array = []

        for index, product in enumerate(products):
            product_data = self.get_dataset_by_extent(
                product,
                product_type=product_type,
                platform=platforms[index] if platforms is not None else None,
                time=time,
                longitude=longitude,
                latitude=latitude,
                measurements=measurements,
                output_crs=output_crs,
                resolution=resolution,
                dask_chunks=dask_chunks)
            if 'time' in product_data:
                product_data['satellite'] = xr.DataArray(
                    np.full(product_data[list(product_data.data_vars)[0]].values.shape, index, dtype="int16"),
                    dims=('time', 'latitude', 'longitude'),
                    coords={
                        'latitude': product_data.latitude,
                        'longitude': product_data.longitude,
                        'time': product_data.time
                    })
                data_array.append(product_data.copy(deep=True))

        data = None
        if len(data_array) > 0:
            combined_data = xr.concat(data_array, 'time')
            data = combined_data.reindex({'time': sorted(combined_data.time.values)})

        return data

    def get_query_metadata(self, product, platform=None, longitude=None, latitude=None, time=None, **kwargs):
        """
        Gets a descriptor based on a request.

        Args:
            platform (string): Platform for which data is requested
            product (string): The name of the product associated with the desired dataset.
            longitude (tuple): Tuple of min,max floats for longitude
            latitude (tuple): Tuple of min,max floats for latitutde
            time (tuple): Tuple of start and end datetimes for requested data
            **kwargs (dict): Keyword arguments for `self.get_dataset_by_extent()`.

        Returns:
            scene_metadata (dict): Dictionary containing a variety of data that can later be
                                   accessed.
        """
        kwargs['measurements'] = []
        dataset = self.get_dataset_by_extent(
            platform=platform, product=product, longitude=longitude,
            latitude=latitude, time=time, **kwargs)

        if len(dataset.dims) == 0:
            return {
                'lat_extents': (None, None),
                'lon_extents': (None, None),
                'time_extents': (None, None),
                'scene_count': 0,
                'pixel_count': 0,
                'tile_count': 0,
                'storage_units': {}
            }

        lon_min, lat_min, lon_max, lat_max = dataset.geobox.extent.envelope
        return {
            'lat_extents': (lat_min, lat_max),
            'lon_extents': (lon_min, lon_max),
            'time_extents': (dataset.time[0].values.astype('M8[ms]').tolist(),
                             dataset.time[-1].values.astype('M8[ms]').tolist()),
            'tile_count':
            dataset.time.size,
            'pixel_count':
            dataset.geobox.shape[0] * dataset.geobox.shape[1],
        }

    def list_acquisition_dates(self, product, platform=None, longitude=None, latitude=None, time=None, **kwargs):
        """
        Get a list of all acquisition dates for a query.

        Args:
            platform (string): Platform for which data is requested
            product (string): The name of the product associated with the desired dataset.
            longitude (tuple): Tuple of min,max floats for longitude
            latitude (tuple): Tuple of min,max floats for latitutde
            time (tuple): Tuple of start and end datetimes for requested data

        Returns:
            times (list): Python list of dates that can be used to query the dc for single time
                          sliced data.
        """
        dataset = self.get_dataset_by_extent(
            product=product, platform=platform, longitude=longitude,
            latitude=latitude, time=time, dask_chunks={}, measurements=[])

        if len(dataset.dims) == 0:
            return []
        return dataset.time.values.astype('M8[ms]').tolist()

    def list_combined_acquisition_dates(self,
                                        products,
                                        platforms=None,
                                        longitude=None,
                                        latitude=None,
                                        time=None,
                                        **kwargs):
        """
        Get a list of all acquisition dates for a query.

        Args:
            platforms (list): Platforms for which data is requested
            products (list): The name of the products associated with the desired dataset.
            longitude (tuple): Tuple of min,max floats for longitude
            latitude (tuple): Tuple of min,max floats for latitutde
            time (tuple): Tuple of start and end datetimes for requested data

        Returns:
            times (list): Python list of dates that can be used to query the dc for single time
                          sliced data.
        """
        dates = []
        for index, product in enumerate(products):
            dataset = self.get_dataset_by_extent(
                product,
                platform=platforms[index] if platforms is not None else None,
                time=time,
                longitude=longitude,
                latitude=latitude,
                dask_chunks={},
                measurements=[])

            if len(dataset.dims) == 0:
                continue

            dates += dataset.time.values.astype('M8[ms]').tolist()

        return dates

    def get_full_dataset_extent(self, product, platform=None, longitude=None, latitude=None, time=None, **kwargs):
        """
        Get a list of all dimensions for a query.

        Args:
            platform (string): Platform for which data is requested
            product (string): The name of the product associated with the desired dataset.
            longitude (tuple): Tuple of min,max floats for longitude
            latitude (tuple): Tuple of min,max floats for latitutde
            time (tuple): Tuple of start and end datetimes for requested data

        Returns:
            dict containing time, latitude, and longitude, each containing the respective xarray dataarray
        """
        dataset = self.get_dataset_by_extent(
            product=product, platform=platform, longitude=longitude,
            latitude=latitude, time=time, dask_chunks={}, measurements=[])

        if len(dataset.dims) == 0:
            return []
        return {'time': dataset.time, 'latitude': dataset.latitude, 'longitude': dataset.longitude}

    def get_datacube_metadata(self, product, platform=None):
        """
        Gets some details on the cube and its contents.

        Args:
            platform (string): Desired platform for requested data.
            product (string): Desired product for requested data.

        Returns:
            datacube_metadata (dict): a dict with multiple keys containing relevant metadata.
        """

        return self.get_query_metadata(product, platform=platform)

    def validate_measurements(self, product, measurements, **kwargs):
        """Ensure that your measurements exist for the product before loading.
        """
        measurement_list = self.dc.list_measurements(with_pandas=False)
        measurements_for_product = filter(lambda x: x['product'] == product, measurement_list)
        valid_measurements_name_array = map(lambda x: x['name'], measurements_for_product)

        return set(measurements).issubset(set(valid_measurements_name_array))
