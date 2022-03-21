import csv
from datetime import datetime, timezone
from dateutil import relativedelta, parser
import os
import fsspec
from datacube import Datacube
from datacube.utils import geometry
import numpy
import rasterio.features
from shapely import geometry as shapely_geom

import logging

logger = logging.getLogger(__name__)


def get_last_date(fpath, max_days=None):
    try:
        current_time = datetime.now(timezone.utc)
        of = fsspec.open(fpath, 'r')
        with of as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            last_date = last_line.split(',')[0]
            start_date = parser.parse(last_date)
            start_date = start_date + relativedelta.relativedelta(days=1)
            if max_days:
                if (current_time - start_date).days > max_days:
                    start_date = current_time - relativedelta.relativedelta(
                        days=max_days)
            str_start_date = start_date.strftime('%Y-%m-%d')
            logger.debug(f'Start date is {str_start_date}')
            return str_start_date
    except:  # noqa: E722
        logger.debug(f'Cannot find last date for {fpath}')
        return None


def wofls_fuser(dest, src):
    where_nodata = (src & 1) == 0
    numpy.copyto(dest, src, where=where_nodata)
    return dest


def get_resolution(wofls):
    """Get the resolution for a WOfLs product."""
    resolutions = {
        'ga_ls_wo_3': (-30, 30),
        'wofs_albers': (-25, 25),
        'ga_s2_wo_3': (-10, 10),
    }
    return resolutions[wofls]


def get_dataset_maturity(wofls):
    """Get the dataset_maturity flag for a WOfLs product."""
    if wofls == 'ga_s2_wo_3':
        return 'interim'
    return None


# Define a function that does all of the work
def generate_wb_timeseries(shapes, config_dict):
    """
    This is where the code processing is actually done. This code takes in a
    polygon, and the and a config dict which contains: shapefile's crs, output
    directory, id_field, time_span, and include_uncertainty which says whether
    to include all data as well as an invalid pixel count which can be used
    for measuring uncertainty performs a polygon drill into the wofs_albers
    product. The resulting xarray, which contains the water classified pixels
    for that polygon over every available timestep, is used to calculate the
    percentage of the water body that is wet at each time step. The outputs
    are written to a csv file named using the polygon UID, which is a geohash
    of the polygon's centre coords.
    Inputs:
    shapes - polygon to be interrogated
    config_dict - many config settings including crs, id_field, time_span,
                  shapefile
    Outputs:
    Nothing is returned from the function, but a csv file is written out to
        disk
    """
    output_dir = config_dict['output_dir']
    crs = config_dict['crs']
    id_field = config_dict['id_field']
    time_span = config_dict['time_span']
    include_uncertainty = config_dict['include_uncertainty']
    wofls = config_dict['wofs_ls_summary_annual']
    assert wofls

    # Some query parameters will be different for different WOfL products.
    output_res = get_resolution(wofls)
    dataset_maturity = get_dataset_maturity(wofls)

    if include_uncertainty:
        unknown_percent_threshold = 100
    else:
        unknown_percent_threshold = 10

    with Datacube(app='Polygon drill') as dc:
        first_geometry = shapes['geometry']

        str_poly_name = shapes['properties'][id_field]

        try:
            fpath = os.path.join(
                output_dir, f'{str_poly_name[0:4]}/{str_poly_name}.csv')
        except TypeError:
            str_poly_name = str(int(str_poly_name)).zfill(6)
            fpath = os.path.join(
                output_dir, f'{str_poly_name[0:4]}/{str_poly_name}.csv')
        geom = geometry.Geometry(first_geometry, crs=crs)
        current_year = datetime.now().year

        if time_span == 'ALL':
            if shapely_geom.shape(first_geometry).envelope.area > 2000000:
                years = range(1986, current_year + 1, 5)
                time_periods = [(str(year), str(year + 4)) for year in years]
            else:
                time_periods = [('1986', str(current_year))]
        elif time_span == 'APPEND':
            start_date = get_last_date(fpath)
            if start_date is None:
                logger.debug(f'There is no csv for {str_poly_name}')
                return 1
            time_periods = [(start_date, str(current_year))]
        elif time_span == 'CUSTOM':
            time_periods = [(config_dict['start_dt'], config_dict['end_date'])]

        valid_capacity_pc = []
        valid_capacity_ct = []
        invalid_capacity_ct = []
        date_list = []
        for time in time_periods:
            wb_capacity_pc = []
            wb_capacity_ct = []
            wb_invalid_ct = []
            dry_observed = []
            invalid_observations = []

            # Set up the query, and load in all of the WOFS layers
            query = {'geopolygon': geom, 'time': time,
                     'output_crs': crs, 'resolution': output_res,
                     'resampling': 'nearest'}
            if dataset_maturity:
                query['dataset_maturity'] = dataset_maturity
            logger.debug('Query: {}'.format({k: v for k, v in query.items()
                                             if k != 'geopolygon'}))
            wofl = dc.load(product=wofs_ls_summary_annual, group_by='solar_day',
                           fuse_func=wofls_fuser, **query)

            if len(wofl.attrs) == 0:
                logger.debug(
                    f'There is no new data for {str_poly_name} in {time}')
                # TODO(MatthewJA): Confirm (with Ness?) that changing this
                # return to a continue doesn't break things.
                continue
            # Make a mask based on the polygon (to remove extra data
            # outside of the polygon)
            mask = rasterio.features.geometry_mask(
                [geom.to_crs(wofl.geobox.crs) for geoms in [geom]],
                out_shape=wofl.geobox.shape,
                transform=wofl.geobox.affine,
                all_touched=False,
                invert=True)
            # mask the data to the shape of the polygon
            # the geometry width and height must both be larger than one pixel
            # to mask.
            if (geom.boundingbox.width > 25.3 and
                    geom.boundingbox.height > 25.3):
                wofl_masked = wofl.water.where(mask)
            else:
                wofl_masked = wofl.water

            # Work out how full the waterbody is at every time step
            for ix, times in enumerate(wofl.time):

                # Grab the data for our timestep
                all_the_bit_flags = wofl_masked.isel(time=ix)

                # Find all the wet/dry pixels for that timestep
                lsa_wet = all_the_bit_flags.where(
                    all_the_bit_flags == 136).count().item()
                lsa_dry = all_the_bit_flags.where(
                    all_the_bit_flags == 8).count().item()
                sea_wet = all_the_bit_flags.where(
                    all_the_bit_flags == 132).count().item()
                sea_dry = all_the_bit_flags.where(
                    all_the_bit_flags == 4).count().item()
                sea_lsa_wet = all_the_bit_flags.where(
                    all_the_bit_flags == 140).count().item()
                sea_lsa_dry = all_the_bit_flags.where(
                    all_the_bit_flags == 12).count().item()
                wet_pixels = (all_the_bit_flags.where(
                    all_the_bit_flags == 128).count().item() +
                    lsa_wet + sea_wet + sea_lsa_wet)
                dry_pixels = (all_the_bit_flags.where(
                    all_the_bit_flags == 0).count().item()
                    + lsa_dry + sea_dry + sea_lsa_dry)

                # Count the number of masked observations
                masked_all = all_the_bit_flags.count().item()
                # Turn our counts into percents
                try:
                    water_percent = round((wet_pixels / masked_all * 100), 1)
                    dry_percent = round((dry_pixels / masked_all * 100), 1)
                    missing_pixels = masked_all - (wet_pixels + dry_pixels)
                    unknown_percent = missing_pixels / masked_all * 100

                except ZeroDivisionError:
                    water_percent = 0.0
                    dry_percent = 0.0
                    unknown_percent = 100.0
                    missing_pixels = masked_all
                    logger.debug(f'{str_poly_name} has divide by zero error')

                # Append the percentages to a list for each timestep
                # Filter out timesteps with < 90% valid observations. Add
                # empty values for timesteps with < 90% valid. if you set
                # 'UNCERTAINTY = True' in your config file then you will
                # only filter out timesteps with 100% invalid pixels.
                # You will also record the number invalid pixels per timestep.

                if unknown_percent < unknown_percent_threshold:
                    wb_capacity_pc.append(water_percent)
                    invalid_observations.append(unknown_percent)
                    wb_invalid_ct.append(missing_pixels)
                    dry_observed.append(dry_percent)
                    wb_capacity_ct.append(wet_pixels)
                else:
                    wb_capacity_pc.append('')
                    invalid_observations.append('')
                    wb_invalid_ct.append('')
                    dry_observed.append('')
                    wb_capacity_ct.append('')

            valid_obs = wofl.time.dropna(dim='time')
            valid_obs = valid_obs.to_dataframe()
            if 'spatial_ref' in valid_obs.columns:
                valid_obs = valid_obs.drop(columns=['spatial_ref'])
            valid_capacity_pc += wb_capacity_pc
            valid_capacity_ct += wb_capacity_ct
            invalid_capacity_ct += wb_invalid_ct
            date_list += valid_obs.to_csv(None, header=False, index=False,
                                          date_format="%Y-%m-%dT%H:%M:%SZ"
                                          ).split('\n')
            date_list.pop()

        if date_list:
            if include_uncertainty:
                rows = zip(date_list, valid_capacity_pc, valid_capacity_ct,
                           invalid_capacity_ct)
            else:
                rows = zip(date_list, valid_capacity_pc, valid_capacity_ct)
            os.makedirs(os.path.dirname
                        (fpath), exist_ok=True)
            if time_span == 'APPEND':
                of = fsspec.open(fpath, 'a')
                with of as f:
                    writer = csv.writer(f)
                    for row in rows:
                        writer.writerow(row)
            else:
                of = fsspec.open(fpath, 'w')
                with of as f:
                    writer = csv.writer(f)
                    headings = ['Observation Date', 'Wet pixel percentage',
                                'Wet pixel count (n = {0})'.format(masked_all)]
                    if include_uncertainty:
                        headings.append('Invalid pixel count')
                    writer.writerow(headings)
                    for row in rows:
                        writer.writerow(row)
        else:
            logger.info(f'{str_poly_name} has no new good valid data')
        return True
    
def get_shapes(config_dict: dict,
               wb_ids: [str] or None,
               id_field: str) -> [dict]:
    import fiona
    output_dir = config_dict['output_dir']

    # If missing_only, remove waterbodies that already exist.
    if config_dict['missing_only']:
        logger.info("Filtering waterbodies with existing outputs")
        # NOTE(MatthewJA): I removed references to processed_file here -
        # I think it should be captured later. If this induces a bug,
        # here's a great place to start looking.
        # TODO(MatthewJA): Use Paths earlier on and don't convert here.
        # TODO(MatthewJA): Why doesn't this break with S3 paths?
        output_dir = Path(config_dict['output_dir'])
        missing_list = []
        for id_ in wb_ids:
            out_path = output_dir / id_[:4] / f'{id}.csv'
            if out_path.exists():
                continue

            missing_list.append(id_)
        wb_ids = missing_list

        logger.info(
            f'{len(missing_list)} missing polygons to process')

    # Filter the list of shapes to include only specified polygons,
    # possibly constrained to a state.
    filtered_shapes = []
    wb_ids = set(wb_ids or [])  # for quick membership lookups
    config_state = config_dict.get('filter_state')
    with fiona.open(config_dict['shape_file']) as shapes:
        for shape in shapes:
            wb_id = shape['properties'][id_field]
            if wb_ids and wb_id not in wb_ids:
                logger.debug(f'Rejecting {wb_id} (not in wb_ids)')
                continue

            if config_state and shape['properties']['STATE'] != config_state:
                logger.debug(
                    f'Rejecting {wb_id} (not in state {config_state})')
                continue

            logger.debug(f'Accepting {wb_id}')
            filtered_shapes.append(shape)

    return filtered_shapes

