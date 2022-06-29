import datacube
import xarray as xr
from deafrica_tools.datahandling import load_ard
from deafrica_tools.bandindices import calculate_indices
from odc.algo import xr_geomedian


# Define functions to load features
def feature_layers(query):

    # Connnect to datacube
    dc = datacube.Datacube(app="crop_type_ml")

#     # ----------------- S2 BIMONTHLY GEOMEDIANS -----------------
#     # These are designed to take the geomedian for every two months,
#     # Starting 6 calendar months before the initial collection date.
#     # This is controlled through the input query

    s2_measurements = [
        "blue",
        "green",
        "red",
        "nir",
        "swir_1",
        "swir_2",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
    ]

    ds = load_ard(
        dc=dc,
        products=["s2_l2a"],
        measurements=s2_measurements,
        group_by="solar_day",
        verbose=False,
        **query,
    )

    # Resample, apply geomedian function
    grouped = ds.resample(time="2MS")
    ds_bimonthly_geomedian = grouped.map(xr_geomedian)

    # Compute Indices
    ds_bimonthly_geomedian = calculate_indices(
        ds_bimonthly_geomedian,
        index=["NDVI", "LAI", "SAVI", "MSAVI", "MNDWI"],
        drop=False,
        collection="s2",
    ).squeeze()

    # Split into multiple datasets, dropping time on each
    ds_bimonthly_geomedian_list = [
        ds_bimonthly_geomedian.isel({"time": i}).reset_coords().drop_vars(["time"])
        for i in range(ds_bimonthly_geomedian.sizes["time"])
    ]

    # Rename
    for i in range(len(ds_bimonthly_geomedian_list)):
        months_prior = 6 - i * 2
        rename_dict = {
            key: f"{key}_geomed_{months_prior}monthsprior"
            for key in list(ds_bimonthly_geomedian_list[i].keys())
        }
        ds_bimonthly_geomedian_list[i] = ds_bimonthly_geomedian_list[i].rename(
            name_dict=rename_dict
        )

    # ----------------- S2 Annual GEOMEDIAN -----------------
    # Load this for the year prior to the data collection

    # Update query to use year prior
    geomad_query = query.copy()
    year_for_annual_geomedian = str(query["time"][1].year - 1)
    geomad_query.update({"time": (year_for_annual_geomedian)})

    # load s2 annual geomedian
    ds_annual_geomedian = dc.load(
        product="gm_s2_annual", measurements=s2_measurements, **geomad_query
    )

    # calculate some band indices
    ds_annual_geomedian = calculate_indices(
        ds_annual_geomedian,
        index=["NDVI", "LAI", "SAVI", "MSAVI", "MNDWI"],
        drop=False,
        collection="s2",
    ).squeeze()

    # Rename
    annual_rename_dict = {
        key: f"{key}_geomed_{year_for_annual_geomedian}annual"
        for key in list(ds_annual_geomedian.keys())
    }
    ds_annual_geomedian = ds_annual_geomedian.rename(name_dict=annual_rename_dict)

#     # ----------------- S1 BIMONTHLY GEOMEDIANS -----------------

    s1_query = query.copy()
    s1_query.update({"sat_orbit_state": "ascending"})

    s1_ds = load_ard(
        dc=dc,
        products=["s1_rtc"],
        measurements=["vv", "vh"],
        group_by="solar_day",
        verbose=False,
        **s1_query,
    )

    # Resample, apply geomedian function, compute result to store in memory
    grouped_s1 = s1_ds.resample(time="2MS")
    ds_s1_bimonthly_geomedian = grouped_s1.map(xr_geomedian)

    # Split into multiple datasets, dropping time on each
    ds_s1_bimonthly_geomedian_list = [
        ds_s1_bimonthly_geomedian.isel({"time": i}).reset_coords().drop_vars(["time"])
        for i in range(ds_s1_bimonthly_geomedian.sizes["time"])
    ]

    # Rename
    for i in range(len(ds_s1_bimonthly_geomedian_list)):
        months_prior = 6 - i * 2
        rename_dict = {
            key: f"{key}_s1_geomed_{months_prior}monthsprior"
            for key in list(ds_s1_bimonthly_geomedian_list[i].keys())
        }
        ds_s1_bimonthly_geomedian_list[i] = ds_s1_bimonthly_geomedian_list[i].rename(
            name_dict=rename_dict
        )

#     # -------- LANDSAT BIMONTHLY FRACTIONAL COVER -----------

    # load all available fc data
    ds_fc = dc.load(product="fc_ls", collection_category="T1", **query)
    
    ds_fc_median = ds_fc.resample(time="2MS").median()
    
    # Split into multiple datasets, dropping time on each
    ds_fc_bimonthly_median_list = [
        ds_fc_median.isel({"time": i}).reset_coords().drop_vars(["time", "spatial_ref"])
        for i in range(ds_fc_median.sizes["time"])
    ]

    # Rename
    for i in range(len(ds_fc_bimonthly_median_list)):
        months_prior = 6 - i * 2
        rename_dict = {
            key: f"{key}_fc_{months_prior}monthsprior"
            for key in list(ds_fc_bimonthly_median_list[i].keys())
        }
        ds_fc_bimonthly_median_list[i] = ds_fc_bimonthly_median_list[i].rename(name_dict=rename_dict)
        
    # ----------------- FINAL MERGED XARRAY -----------------

    # Create a list to keep all items for final merge
    ds_list = []
    ds_list.extend(ds_bimonthly_geomedian_list)
    ds_list.append(ds_annual_geomedian)
    ds_list.extend(ds_s1_bimonthly_geomedian_list)
    ds_list.extend(ds_fc_bimonthly_median_list)

    ds_final = xr.merge(ds_list)

    return ds_final


# # Define functions to load features
# def feature_layers(query):

#     # Connnect to datacube
#     dc = datacube.Datacube(app="crop_type_ml")

# #     # ----------------- S2 BIMONTHLY GEOMEDIANS -----------------
# #     # These are designed to take the geomedian for every two months,
# #     # Starting 6 calendar months before the initial collection date.
# #     # This is controlled through the input query

#     s2_measurements = [
#         "blue",
#         "green",
#         "red",
#         "nir",
#         "swir_1",
#         "swir_2",
#         "red_edge_1",
#         "red_edge_2",
#         "red_edge_3",
#     ]

# #     ds = dc.load(
# #         product="s2_l2a",
# #         measurements=s2_measurements,
# #         **query,
# #     )

#     ds = load_ard(
#         dc=dc,
#         products=["s2_l2a"],
#         measurements=s2_measurements,
#         group_by="solar_day",
#         verbose=False,
#         **query,
#     )

#     # Resample, apply geomedian function
#     grouped = ds.resample(time="2MS")
#     ds_bimonthly_geomedian = grouped.map(xr_geomedian)

#     # Compute Indices
#     ds = calculate_indices(
#         ds,
#         index=["NDVI", "LAI", "SAVI", "MSAVI", "MNDWI"],
#         drop=False,
#         collection="s2",
#     ).squeeze()

#     # Split into multiple datasets, dropping time on each
#     ds_bimonthly_geomedian_list = [
#         ds_bimonthly_geomedian.isel({"time": i}).reset_coords().drop_vars(["time"])
#         for i in range(ds_bimonthly_geomedian.sizes["time"])
#     ]

#     # Rename
#     for i in range(len(ds_bimonthly_geomedian_list)):
#         months_prior = 6 - i * 2
#         rename_dict = {
#             key: f"{key}_geomed_{months_prior}monthsprior"
#             for key in list(ds_bimonthly_geomedian_list[i].keys())
#         }
#         ds_bimonthly_geomedian_list[i] = ds_bimonthly_geomedian_list[i].rename(
#             name_dict=rename_dict
#         )

#     # ----------------- S2 Annual GEOMEDIAN -----------------
#     # Load this for the year prior to the data collection

#     # Update query to use year prior
#     geomad_query = query.copy()
#     year_for_annual_geomedian = str(query["time"][1].year - 1)
#     geomad_query.update({"time": (year_for_annual_geomedian)})

#     # load s2 annual geomedian
#     ds_annual_geomedian = dc.load(
#         product="gm_s2_annual", measurements=s2_measurements, **geomad_query
#     )

#     # calculate some band indices
#     ds_annual_geomedian = calculate_indices(
#         ds_annual_geomedian,
#         index=["NDVI", "LAI", "SAVI", "MSAVI", "MNDWI"],
#         drop=False,
#         collection="s2",
#     ).squeeze()

#     # Rename
#     annual_rename_dict = {
#         key: f"{key}_geomed_{year_for_annual_geomedian}annual"
#         for key in list(ds_annual_geomedian.keys())
#     }
#     ds_annual_geomedian = ds_annual_geomedian.rename(name_dict=annual_rename_dict)

# #     # ----------------- S1 BIMONTHLY GEOMEDIANS -----------------

#     s1_query = query.copy()
#     s1_query.update({"sat_orbit_state": "ascending"})

#     s1_ds = load_ard(
#         dc=dc,
#         products=["s1_rtc"],
#         measurements=["vv", "vh"],
#         group_by="solar_day",
#         verbose=False,
#         **s1_query,
#     )

#     # Resample, apply geomedian function, compute result to store in memory
#     grouped_s1 = s1_ds.resample(time="2MS")
#     ds_s1_bimonthly_geomedian = grouped_s1.map(xr_geomedian)

#     # Split into multiple datasets, dropping time on each
#     ds_s1_bimonthly_geomedian_list = [
#         ds_s1_bimonthly_geomedian.isel({"time": i}).reset_coords().drop_vars(["time"])
#         for i in range(ds_s1_bimonthly_geomedian.sizes["time"])
#     ]

#     # Rename
#     for i in range(len(ds_s1_bimonthly_geomedian_list)):
#         months_prior = 6 - i * 2
#         rename_dict = {
#             key: f"{key}_s1_geomed_{months_prior}monthsprior"
#             for key in list(ds_s1_bimonthly_geomedian_list[i].keys())
#         }
#         ds_s1_bimonthly_geomedian_list[i] = ds_s1_bimonthly_geomedian_list[i].rename(
#             name_dict=rename_dict
#         )

# #     # ----------------- MONTHLY RAINFALL --------------------

# #     ds_rf_month = dc.load(
# #         product="rainfall_chirps_monthly",
# #         **query,
# #     )

# #     # Split into multiple datasets, dropping time on each
# #     ds_rf_monthly_list = [
# #         ds_rf_month.isel({"time": i}).reset_coords().drop_vars(["time", "spatial_ref"])
# #         for i in range(ds_rf_month.sizes["time"])
# #     ]

# #     # Rename
# #     for i in range(len(ds_rf_monthly_list)):
# #         months_prior = 6 - i * 1
# #         rename_dict = {
# #             key: f"{key}_{months_prior}monthsprior"
# #             for key in list(ds_rf_monthly_list[i].keys())
# #         }
# #         ds_rf_monthly_list[i] = ds_rf_monthly_list[i].rename(name_dict=rename_dict)

# #     # -------- LANDSAT BIMONTHLY FRACTIONAL COVER -----------

#     # load all available fc data
#     ds_fc = dc.load(product="fc_ls", collection_category="T1", **query)
    
#     ds_fc_median = ds_fc.resample(time="2MS").median()
    
#     # Split into multiple datasets, dropping time on each
#     ds_fc_bimonthly_median_list = [
#         ds_fc_median.isel({"time": i}).reset_coords().drop_vars(["time", "spatial_ref"])
#         for i in range(ds_fc_median.sizes["time"])
#     ]

#     # Rename
#     for i in range(len(ds_fc_bimonthly_median_list)):
#         months_prior = 6 - i * 2
#         rename_dict = {
#             key: f"{key}_fc_{months_prior}monthsprior"
#             for key in list(ds_fc_bimonthly_median_list[i].keys())
#         }
#         ds_fc_bimonthly_median_list[i] = ds_fc_bimonthly_median_list[i].rename(name_dict=rename_dict)
        
#     # ----------------- FINAL MERGED XARRAY -----------------

#     # Create a list to keep all items for final merge
#     ds_list = []
#     ds_list.extend(ds_bimonthly_geomedian_list)
#     ds_list.append(ds_annual_geomedian)
#     ds_list.extend(ds_s1_bimonthly_geomedian_list)
# #     ds_list.extend(ds_rf_monthly_list)
#     ds_list.extend(ds_fc_bimonthly_median_list)

#     ds_final = xr.merge(ds_list)

#     return ds_final