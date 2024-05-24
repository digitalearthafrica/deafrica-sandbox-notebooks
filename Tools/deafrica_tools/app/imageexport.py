"""
Create an interactive map for selecting satellite imagery and exporting image files.
"""

# Load modules
import datacube
import itertools
import numpy as np
import matplotlib.pyplot as plt
from odc.ui import select_on_a_map
from datacube.utils.geometry import CRS
from datacube.utils import masking
from skimage import exposure
from ipyleaflet import (WMSLayer, basemaps, basemap_to_tiles)
from traitlets import Unicode

from deafrica_tools.spatial import reverse_geocode
from deafrica_tools.dask import create_local_dask_cluster


def select_region_app(date,
                      satellites,
                      size_limit=10000):
    """
    An interactive app that allows the user to select a region from a
    map using imagery from Sentinel-2 and Landsat. The output of this
    function is used as the input to :func:`export_image_app` to export high-
    resolution satellite images.

    Last modified: September 2021

    Parameters
    ----------
    date : str
        The exact date used to plot imagery on the interactive map
        (e.g. ``date='1988-01-01'``).
    satellites : str
        The satellite data to plot on the interactive map. The 
        following options are supported:

            ``'Landsat-9'``: data from the Landsat 9 satellite
            ``'Landsat-8'``: data from the Landsat 8 satellite
            ``'Landsat-7'``: data from the Landsat 7 satellite
            ``'Landsat-5'``: data from the Landsat 5 satellite
            ``'Sentinel-2'``: data from Sentinel-2A and Sentinel-2B
            ``'Sentinel-2 geomedian'``: data from the Sentinel-2 annual geomedian

    size_limit : int, optional
        An optional size limit for the area selection in sq km.
        Defaults to 10000 sq km.

    Returns
    -------
    A dictionary containing:

        * 'geopolygon' (defining the area to export imagery from),
        * 'date' (date used to export imagery), and
        * 'satellites' (the satellites from which to extract imagery).

    These are passed to the :func:`export_image_app` function to export the image.
    """

    ########################
    # Select and load data #
    ########################

    # Load DEA WMS
    class TimeWMSLayer(WMSLayer):
        time = Unicode('').tag(sync=True, o=True)

    # WMS layers
    wms_params = {
        'Landsat-9': 'ls9_sr',
        'Landsat-8': 'ls8_sr',
        'Landsat-7': 'ls7_sr',
        'Landsat-5': 'ls5_sr',
        'Sentinel-2': 's2_l2a',
        'Sentinel-2 geomedian': 'gm_s2_annual'
    }

    time_wms = TimeWMSLayer(url='https://ows.digitalearth.africa/',
                            layers=wms_params[satellites],
                            time=date,
                            format='image/png',
                            transparent=True,
                            attribution='Digital Earth Africa')

    # Plot interactive map to select area
    basemap = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    geopolygon = select_on_a_map(height='1000px',
                                 layers=(
                                     basemap,
                                     time_wms,
                                 ),
                                 center=(4, 20),
                                 zoom=4)

    # Test size of selected area
    area = geopolygon.to_crs(crs=CRS('epsg:6933')).area / 1000000
    if area > size_limit:
        print(f'Warning: Your selected area is {area:.00f} sq km. '
              f'Please select an area of less than {size_limit} sq km.'
              f'\nTo select a smaller area, re-run the cell '
              f'above and draw a new polygon.')

    else:
        return {'geopolygon': geopolygon,
                'date': date,
                'satellites': satellites}


def export_image_app(geopolygon,
                     date,
                     satellites,
                     style='True colour',
                     resolution=None,
                     vmin=0,
                     vmax=2000,
                     percentile_stretch=None,
                     power=None,
                     image_proc_funcs=None,
                     output_format="jpg",
                     standardise_name=False):
    """
    Exports Digital Earth Africa satellite data as an image file
    based on the extent and time period selected using
    :func:`select_region_app`. The function supports Sentinel-2 and Landsat
    data, creating True and False colour images.

    By default, files are named using:

        ``"<product> - <YYYY-MM-DD> - <site, state> - <description>.png"``

    Set ``standardise_name=True`` for a machine-readable name:

        ``"<product>_<YYYY-MM-DD>_<site-state>_<description>.png"``

    Last modified: September 2021

    Parameters
    ----------
    geopolygon : datacube.utils.geometry object
        A datacube geopolygon providing the spatial bounds used to load
        satellite data.
    date : str
        The exact date used to extract imagery
        (e.g. `date='1988-01-01'`).
    satellites : str
        The satellite data to be used to extract imagery. The 
        following options are supported:

            ``'Landsat-9'``: data from the Landsat 9 satellite
            ``'Landsat-8'``: data from the Landsat 8 satellite
            ``'Landsat-7'``: data from the Landsat 7 satellite
            ``'Landsat-5'``: data from the Landsat 5 satellite
            ``'Sentinel-2'``: data from Sentinel-2A and Sentinel-2B
            ``'Sentinel-2 geomedian'``: data from the Sentinel-2 annual geomedian

    style : str, optional
        The style used to produce the image. Two options are currently
        supported:

            * ``'True colour'``: Creates a true colour image using the red,
              green and blue satellite bands
            * ``'False colour'``: Creates a false colour image using
              short-wave infrared, infrared and green satellite bands.
              The specific bands used vary between Landsat and Sentinel-2.

    resolution : tuple, optional
        The spatial resolution to load data. By default, the tool will
        automatically set the best possible resolution depending on the
        satellites selected (i.e 30 m for Landsat, 10 m for Sentinel-2).
        Increasing this (e.g. to ``resolution=(-100, 100)``) can be useful
        for loading large spatial extents.
    vmin, vmax : int or float
        The minimum and maximum surface reflectance values used to
        clip the resulting imagery to enhance contrast.
    percentile_stretch : tuple of floats, optional
        An tuple of two floats (between 0.00 and 1.00) that can be used
        to clip the imagery to based on percentiles to get more control
        over the brightness and contrast of the image. The default is
        ``None``; ``(0.02, 0.98)`` is equivelent to ``robust=True``. If this
        parameter is used, ``vmin`` and ``vmax`` will have no effect.
    power : float, optional
        Raises imagery by a power to reduce bright features and
        enhance dark features. This can add extra definition over areas
        with extremely bright features like snow, beaches or salt pans.
    image_proc_funcs : list of funcs, optional
        An optional list containing functions that will be applied to
        the output image. This can include image processing functions
        such as increasing contrast, unsharp masking, saturation etc.
        The function should take AND return a `numpy.ndarray` with
        shape ``[y, x, bands]``. If your function has parameters, you
        can pass in custom values using a lambda function, e.g.:
        ``[lambda x: skimage.filters.unsharp_mask(x, radius=5, amount=0.2)]``
    output_format : str, optional
        The output file format of the image. Valid options include ``'jpg'``
        and ``'png'``. Defaults to ``'jpg'``.
    standardise_name : bool, optional
        Whether to export the image file with a machine-readable
        file name (e.g. ``<product>_<YYYY-MM-DD>_<site-state>_<description>.png``)
    """

    ###########################
    # Set up satellite params #
    ###########################

    sat_params = {
        'Landsat-9': {
            'products': ['ls9_sr'],
            'resolution': [-30, 30],
            'styles': {
                'True colour': ['red', 'green', 'blue'],
                'False colour': ['swir_1', 'nir', 'green']
            }
        },
        'Landsat-8': {
            'products': ['ls8_sr'],
            'resolution': [-30, 30],
            'styles': {
                'True colour': ['red', 'green', 'blue'],
                'False colour': ['swir_1', 'nir', 'green']
            }
        },
        'Landsat-7': {
            'products': ['ls7_sr'],
            'resolution': [-30, 30],
            'styles': {
                'True colour': ['red', 'green', 'blue'],
                'False colour': ['swir_1', 'nir', 'green']
            }
        },
        'Landsat-5': {
            'products': ['ls5_sr'],
            'resolution': [-30, 30],
            'styles': {
                'True colour': ['red', 'green', 'blue'],
                'False colour': ['swir_1', 'nir', 'green']
            }
        },
        'Sentinel-2': {
            'products': ['s2_l2a'],
            'resolution': [-10, 10],
            'styles': {
                'True colour': ['red', 'green', 'blue'],
                'False colour': ['swir_2', 'nir_1', 'green']
            }
        },
        'Sentinel-2 geomedian': {
            'products': ['gm_s2_annual'],
            'resolution': [-10, 10],
            'styles': {
                'True colour': ['red', 'green', 'blue'],
                'False colour': ['swir_2', 'nir_1', 'green']
            }
        },
    }

    #############
    # Load data #
    #############

    # Connect to datacube database
    dc = datacube.Datacube(app='Exporting_satellite_images')

    # Configure local dask cluster
    client = create_local_dask_cluster(return_client=True)

    # Create query after adjusting interval time to UTC by
    # adding a UTC offset of -10 hours.
    start_date = np.datetime64(date)
    query_params = {
        'time': (str(start_date)),
        'geopolygon': geopolygon
    }

    # Find matching datasets
    dss = [
        dc.find_datasets(product=i, **query_params)
        for i in sat_params[satellites]['products']
    ]
    dss = list(itertools.chain.from_iterable(dss))

    # Get CRS and sensor
    crs = str(dss[0].crs)

    if satellites == 'Sentinel-2 geomedian':
        sensor = satellites
    else:
        sensor = dss[0].metadata_doc['properties']['eo:platform'].capitalize()
        sensor = sensor[0:-1].replace('_', '-') + sensor[-1].capitalize()

    # Use resolution if provided, otherwise use default
    if resolution:
        sat_params[satellites]['resolution'] = resolution

    load_params = {
        'output_crs': crs,
        'resolution': sat_params[satellites]['resolution'],
        'resampling': 'bilinear'
    }

    # Load data from datasets
    ds = dc.load(datasets=dss,
                 measurements=sat_params[satellites]['styles'][style],
                 group_by='solar_day',
                 dask_chunks={
                     'time': 1,
                     'x': 3000,
                     'y': 3000
                 },
                 **load_params,
                 **query_params)
    ds = masking.mask_invalid_data(ds)

    rgb_array = ds.isel(time=0).to_array().values

    ############
    # Plotting #
    ############

    # Create unique file name
    centre_coords = geopolygon.centroid.coords[0][::-1]
    site = reverse_geocode(coords=centre_coords)
    fname = (f"{sensor} - {date} - {site} - {style}, "
             f"{load_params['resolution'][1]} m resolution.{output_format}")

    # Remove spaces and commas if requested
    if standardise_name:
        fname = fname.replace(' - ', '_').replace(', ',
                                                  '-').replace(' ',
                                                               '-').lower()

    print(
        f'\nExporting image to {fname}.\nThis may take several minutes to complete...'
    )

    # Convert to numpy array
    rgb_array = np.transpose(rgb_array, axes=[1, 2, 0])

    # If percentile stretch is supplied, calculate vmin and vmax
    # from percentiles
    if percentile_stretch:
        vmin, vmax = np.nanpercentile(rgb_array, percentile_stretch)

    # Raise by power to dampen bright features and enhance dark.
    # Raise vmin and vmax by same amount to ensure proper stretch
    if power:
        rgb_array = rgb_array**power
        vmin, vmax = vmin**power, vmax**power

    # Rescale/stretch imagery between vmin and vmax
    rgb_rescaled = exposure.rescale_intensity(rgb_array.astype(float),
                                              in_range=(vmin, vmax),
                                              out_range=(0.0, 1.0))

    # Apply image processing funcs
    if image_proc_funcs:
        for i, func in enumerate(image_proc_funcs):
            print(f'Applying custom function {i + 1}')
            rgb_rescaled = func(rgb_rescaled)

    # Plot RGB
    plt.imshow(rgb_rescaled)

    # Export to file
    plt.imsave(fname=fname, arr=rgb_rescaled, format=output_format)

    # Close dask client
    client.shutdown()

    print('Finished exporting image.')
