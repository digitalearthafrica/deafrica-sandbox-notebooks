import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

def create_geometries_from_manual_edit(df, geom_col):

    # Raise a KeyError if geom_col does not appear in the dataframe
    if geom_col not in df.columns:
        raise KeyError(f"{geom_col} is not a valid column of this dataframe.")

    # Convert from string `lat,lon` to individual columns for lat and lon
    df[["point_lat", "point_lon"]] = df[geom_col].str.split(",", expand=True, n=2)
    
    # Add a point location column, specify as manual to differ from ODK original values
    df.loc[:, "point_location"] = "manual"

    # Convert to geopandas with point geometry
    geom = gpd.points_from_xy(x=df.point_lon, y=df.point_lat, crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df, geometry=geom)

    return gdf


def create_geometries_from_ODK(df):

    # identify rows with either center measurement or corner measurement
    corner_measurement = df.loc[:, "access_consent"] == "no"
    center_measurement = df.loc[:, "access_consent"] == "yes"

    # create a new column to specify location type
    df.loc[corner_measurement, "point_location"] = "outside_corner"
    df.loc[center_measurement, "point_location"] = "center"

    # create a column to store the lat,lon values
    df.loc[corner_measurement, "point_latlon"] = df.loc[
        corner_measurement, "field_outside_corner"
    ]
    df.loc[center_measurement, "point_latlon"] = df.loc[
        center_measurement, "field_center"
    ]

    # Convert from string `lat,lon` to individual columns for lat and lon
    point_latlon_df = pd.DataFrame(
        df["point_latlon"].str.split(",").to_list(), columns=["point_lat", "point_lon"]
    )
    df = pd.concat((df, point_latlon_df), axis="columns")

    # Add boundaries for any locations with center points
    df.loc[center_measurement, "field_boundary"] = df.loc[
        center_measurement, "field_boundary"
    ].str.split("; ")

    # Separate out items where boundary exists to perform next steps
    df_withboundary = df.loc[center_measurement, ("KEY", "field_boundary")]

    # Explode all boundaries to get one row per point in boundary, then split into lat, lon, altitude and accuracy
    df_withboundary = df_withboundary.explode("field_boundary")
    df_withboundary[["lat", "lon", "alt", "acc"]] = df_withboundary[
        "field_boundary"
    ].str.split(" ", expand=True, n=4)
    df_withboundary = df_withboundary.drop(
        ["field_boundary", "alt", "acc"], axis="columns"
    )

    # Convert into Polygons using groupby
    df_withboundary["field_boundary_polygon"] = df_withboundary.groupby(
        df_withboundary.index
    ).apply(lambda g: Polygon(gpd.points_from_xy(g["lon"], g["lat"])))

    # Drop duplicates
    df_withboundary = df_withboundary.drop(["lat", "lon"], axis="columns")
    df_withboundary = df_withboundary.drop_duplicates(subset="KEY")

    df = df.set_index("KEY").join(df_withboundary.set_index("KEY"), how="left")

    # Convert to geopandas with point geometry
    geom = gpd.points_from_xy(x=df.point_lon, y=df.point_lat, crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df, geometry=geom)

    return gdf