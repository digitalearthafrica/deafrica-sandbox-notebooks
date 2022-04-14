# Pandas random sampling function
import pandas as pd
import numpy as np
import geopandas as gpd

def stratified_random_sampling(
    da,
    n,
    min_sample_n=5,
    min_threshold_prop=0.05,
    n_strategies=1,
    manual_class_ratios=None,
    out_fname=None,
):
    
    """
    Creates randomly sampled points for post-classification
    accuracy assessment.

    Params:
    -------
    da: xarray.DataArray
        A classified 2-dimensional xarray.DataArray
    n: int
        Total number of points to sample. Ignored if providing
        a dictionary of {class:numofpoints} to 'manual_class_ratios'
    min_sample_n: int
        Number of samples to collect if proportional number is smaller than this
    out_fname: str
        If providing a filepath name, e.g 'sample_points.shp', the
        function will export a shapefile/geojson of the sampling
        points to file.

    Output
    ------
    GeoPandas.Dataframe

    """
    
    # open the dataset as a pandas dataframe
    da = da.squeeze()
    df = da.to_dataframe(name="class")

    # list to store points
    samples = []

    # determine class ratios in image
    class_ratio = pd.DataFrame(
        {
            "proportion": df["class"].value_counts(normalize=True),
            "class": df["class"].value_counts(normalize=True).keys(),
        }
    )

    for _class in class_ratio["class"]:

        class_proportion = class_ratio[class_ratio["class"] == _class][
            "proportion"
        ].values[0]

        if class_proportion >= min_threshold_prop:
            # use relative proportions of classes to sample df
            no_of_points = n * class_proportion

            # If no_of_points is less than the minimum sample number, use minimum sample number instead
            no_of_points = max(min_sample_n, no_of_points)

        else:
            class_proportion = 0
            
        int_no_of_points = int(round(no_of_points))

        # random sample each class
        print(
            "Class "
            + str(_class)
            + ": sampling at "
            + str(int_no_of_points)
            + " coordinates"
        )
        sample_loc = df[df["class"] == _class].sample(n=int_no_of_points * n_strategies)
        
        # create blank columns to be filled
        sample_loc["strategy"] = None
        sample_loc["sample_no"] = None
        
        # Assign sample number
        for i in range(n_strategies):

            strategy_range = slice(i*int_no_of_points, (i+1)*int_no_of_points, 1)

            sample_loc.iloc[strategy_range, sample_loc.columns.get_loc("strategy")] = f"draw_{i}"
            sample_loc.iloc[strategy_range,  sample_loc.columns.get_loc("sample_no")] = np.arange(sample_loc.iloc[strategy_range].shape[0])
        
        samples.append(sample_loc)
                
    all_samples = pd.concat([samples[i] for i in range(0, len(samples))])

    # get pd.mulitindex coords as list
    y = [i[0] for i in list(all_samples.index)]
    x = [i[1] for i in list(all_samples.index)]

    # create geopandas dataframe
    gdf = gpd.GeoDataFrame(
        all_samples, crs=da.crs, geometry=gpd.points_from_xy(x, y)
    ).reset_index()

    #drop current x,y as these are still in existing EPSG. Want to convert to lat/lon
    gdf = gdf.drop(["x", "y"], axis=1)
    
    # Convert to lat/lon
    gdf = gdf.to_crs("EPSG:4326")
    
    # Get back x and y for later use 
    gdf['lat'] = gdf.geometry.x
    gdf['lon'] = gdf.geometry.y

    if out_fname is not None:
        gdf.to_file(out_fname)

    return gdf