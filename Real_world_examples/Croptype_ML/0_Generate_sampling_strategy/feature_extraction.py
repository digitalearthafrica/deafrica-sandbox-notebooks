import datacube
import xarray as xr
from deafrica_tools.bandindices import calculate_indices

# Define functions to load features
def feature_layers(query):

    # Connnect to datacube
    dc = datacube.Datacube(app="crop_type_ml")

    ## S2 Measurements
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
        "BCMAD",
        "EMAD",
        "SMAD",
    ]

    # ----------------- S2 GEOMEDIAN -----------------

    # load s2 annual geomedian
    ds_geomad = dc.load(product="gm_s2_annual", measurements=s2_measurements, **query)

    # calculate some band indices
    ds_geomad = calculate_indices(
        ds_geomad,
        index=["NDVI", "LAI", "SAVI", "MSAVI", "MNDWI"],
        drop=False,
        collection="s2",
    ).squeeze()

    # ----------------- S2 SEMI-ANNUAL GEOMAD -----------------
    ds_semiannual_geomad = dc.load(
        product="gm_s2_semiannual",
        measurements=s2_measurements,
        **query,
    )

    # Split into two seperate datasets
    ds_janjun_geomad = (
        ds_semiannual_geomad.isel(time=0)
        .reset_coords()
        .drop_vars(["time", "spatial_ref"])
    )
    ds_juldec_geomad = (
        ds_semiannual_geomad.isel(time=1)
        .reset_coords()
        .drop_vars(["time", "spatial_ref"])
    )

    # Calculate Indices
    ds_janjun_geomad = calculate_indices(
        ds_janjun_geomad,
        index=["NDVI"],
        drop=False,
        collection="s2",
    ).squeeze()

    ds_juldec_geomad = calculate_indices(
        ds_juldec_geomad,
        index=["NDVI"],
        drop=False,
        collection="s2",
    ).squeeze()

    # Rename
    janjun_rename_dict = {key: f"janjun_{key}" for key in list(ds_janjun_geomad.keys())}
    ds_janjun_geomad = ds_janjun_geomad.rename(name_dict=janjun_rename_dict)

    juldec_rename_dict = {key: f"juldec_{key}" for key in list(ds_juldec_geomad.keys())}
    ds_juldec_geomad = ds_juldec_geomad.rename(name_dict=juldec_rename_dict)

    # ----------------- S1 RADAR -----------------
    ds_S1 = dc.load(
        product="s1_rtc",
        measurements=["vv", "vh", "mask"],
        group_by="solar_day",
        sat_orbit_state="ascending",
        **query,
    )

    mean_vv = ds_S1.vv.where(ds_S1.mask == 1).mean(dim="time")
    mean_vh = ds_S1.vh.where(ds_S1.mask == 1).mean(dim="time")

    ds_S1_mean = xr.merge([mean_vv, mean_vh])

    # ----------------- Fractional Cover Summaries -----------------
    fc_measurements = [
        "pv_pc_90",
        "npv_pc_90",
        "bs_pc_90",
    ]

    ds_annual_fc = dc.load(
        product="fc_ls_summary_annual",
        measurements=fc_measurements,
        **query,
    )

    # ----------------- MERGE -----------------

    ds_final = xr.merge(
        [ds_geomad, ds_janjun_geomad, ds_juldec_geomad, ds_S1_mean, ds_annual_fc]
    )

    return ds_final