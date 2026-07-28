import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# -----------------------------
# Constants
# -----------------------------
G = 9.80665          # gravitational acceleration, m/s²
M_CH4 = 0.01604      # molar mass of methane, kg/mol
M_AIR = 0.02897      # molar mass of dry air, kg/mol
R_EARTH = 6371000    # Earth radius, metres


# -----------------------------
# Helper 1: Rename latitude/longitude
# -----------------------------
def rename_latlon(ds):
    """
    Rename latitude/longitude dimensions or coordinates to lat/lon.
    """
    rename_dict = {}

    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"

    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"

    return ds.rename(rename_dict) if rename_dict else ds


# -----------------------------
# Helper 2: Select DataArray safely
# -----------------------------
def ensure_dataarray(data, variable_name=None):
    """
    Ensure input is an xarray.DataArray.

    If a Dataset is provided, variable_name must be supplied.
    """
    if isinstance(data, xr.DataArray):
        return data

    if isinstance(data, xr.Dataset):
        if variable_name is None:
            raise TypeError(
                "Input is an xarray.Dataset. Please select a variable first, "
                "for example: ds['methane_mixing_ratio']"
            )

        if variable_name not in data:
            raise KeyError(f"Variable '{variable_name}' was not found in the Dataset.")

        return data[variable_name]

    raise TypeError("Input must be an xarray.DataArray or xarray.Dataset.")


# -----------------------------
# Helper 3: Pixel area
# -----------------------------
def gridcell_area_m2(lat, lon):
    """
    Calculate approximate area of each lat/lon grid cell in m².

    This version uses cell edges, which is more robust than using only np.gradient.
    Returns an xarray.DataArray with dims ('lat', 'lon').
    """

    lat_values = np.asarray(lat.values)
    lon_values = np.asarray(lon.values)

    if lat_values.ndim != 1 or lon_values.ndim != 1:
        raise ValueError("lat and lon must be 1D coordinate arrays.")

    # Convert centre coordinates to radians
    lat_rad = np.deg2rad(lat_values)
    lon_rad = np.deg2rad(lon_values)

    # Build coordinate edges
    lat_edges = np.empty(lat_values.size + 1)
    lon_edges = np.empty(lon_values.size + 1)

    lat_edges[1:-1] = (lat_rad[:-1] + lat_rad[1:]) / 2
    lon_edges[1:-1] = (lon_rad[:-1] + lon_rad[1:]) / 2

    lat_edges[0] = lat_rad[0] - (lat_rad[1] - lat_rad[0]) / 2
    lat_edges[-1] = lat_rad[-1] + (lat_rad[-1] - lat_rad[-2]) / 2

    lon_edges[0] = lon_rad[0] - (lon_rad[1] - lon_rad[0]) / 2
    lon_edges[-1] = lon_rad[-1] + (lon_rad[-1] - lon_rad[-2]) / 2

    # Ensure positive cell width
    dlon = np.abs(np.diff(lon_edges))

    # Spherical area formula
    area = (
        R_EARTH ** 2
        * np.abs(np.diff(np.sin(lat_edges)))[:, None]
        * dlon[None, :]
    )

    return xr.DataArray(
        area,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="pixel_area_m2",
        attrs={
            "units": "m2",
            "long_name": "Pixel area"
        },
    )

def convert_ch4(
    ch4_ppb,
    surface_pressure,
    area_m2=None,
    background="quantile",
    background_quantile=0.05,
    spatial_dims=("lat", "lon"),
    clip_negative=True,
    valid_mask=None,
):
    """
    Convert Sentinel-5P TROPOMI methane XCH4 from ppb into methane mass products.

    Outputs include:
      - background methane concentration in ppb
      - methane enhancement in ppb
      - methane column amount in mol/m²
      - methane mass per unit area in kg/m²
      - methane mass per unit area in g/m²
      - methane mass per unit area in tonnes/km²
      - methane mass per pixel in tonnes
      - total methane mass over the AOI in tonnes
      - total methane mass over the AOI in kg
      - mean methane mass per unit area
      - valid observation area

    Parameters
    ----------
    ch4_ppb : xarray.DataArray
        Sentinel-5P methane XCH4 in ppb.

    surface_pressure : xarray.DataArray
        Surface pressure in Pascals.

    area_m2 : xarray.DataArray, optional
        Pixel area in m². If None, it will be calculated from lat/lon.

    background : "quantile", None, float, int, or xarray.DataArray
        Background methane option.

        - "quantile": subtract spatial quantile background.
        - None: no background removed. Converts total methane column.
        - number: subtract fixed background in ppb.
        - DataArray: subtract custom background.

    background_quantile : float
        Quantile used when background="quantile".
        Example: 0.05 means 5th percentile.

    spatial_dims : tuple
        Spatial dimensions, usually ("lat", "lon").

    clip_negative : bool
        If True, negative methane enhancement values are set to zero.

    valid_mask : xarray.DataArray, optional
        Boolean mask where True means valid pixel.
        Use this for cloud mask, QA mask, or combined masks.

    Returns
    -------
    xarray.Dataset
        Methane mass conversion products.
    """

    # Standardize coordinate names
    ch4_ppb = rename_latlon(ch4_ppb)
    surface_pressure = rename_latlon(surface_pressure)

    if area_m2 is not None:
        area_m2 = rename_latlon(area_m2)

    if valid_mask is not None:
        valid_mask = rename_latlon(valid_mask)

    # Check input
    if not isinstance(ch4_ppb, xr.DataArray):
        raise TypeError("ch4_ppb must be an xarray.DataArray.")

    if not isinstance(surface_pressure, xr.DataArray):
        raise TypeError("surface_pressure must be an xarray.DataArray.")

    for dim in spatial_dims:
        if dim not in ch4_ppb.dims:
            raise ValueError(f"Spatial dimension '{dim}' was not found in ch4_ppb.")

    # Automatically calculate pixel area if not supplied
    if area_m2 is None:
        area_m2 = gridcell_area_m2(ch4_ppb["lat"], ch4_ppb["lon"])

    # Align datasets
    ch4_ppb, surface_pressure, area_m2 = xr.align(
        ch4_ppb,
        surface_pressure,
        area_m2,
        join="inner"
    )

    # Apply valid/cloud/QA mask if supplied
    if valid_mask is not None:
        valid_mask = valid_mask.broadcast_like(ch4_ppb)
        ch4_ppb = ch4_ppb.where(valid_mask)
        surface_pressure = surface_pressure.where(valid_mask)

    # Remove invalid pressure values
    surface_pressure = surface_pressure.where(surface_pressure > 0)

    # -----------------------------
    # Background calculation
    # -----------------------------
    if background is None:
        background_ppb = xr.zeros_like(ch4_ppb)
        delta_ppb = ch4_ppb
        mass_type = "total_column"

    elif isinstance(background, str) and background.lower() == "quantile":
        background_ppb = ch4_ppb.quantile(
            background_quantile,
            dim=spatial_dims,
            skipna=True
        )

        delta_ppb = ch4_ppb - background_ppb
        mass_type = f"enhancement_above_{background_quantile}_quantile_background"

    elif np.isscalar(background):
        background_ppb = xr.full_like(ch4_ppb, float(background))
        delta_ppb = ch4_ppb - background_ppb
        mass_type = f"enhancement_above_fixed_{float(background)}_ppb_background"

    elif isinstance(background, xr.DataArray):
        background = rename_latlon(background)
        ch4_ppb, background = xr.align(ch4_ppb, background, join="inner")
        delta_ppb = ch4_ppb - background
        background_ppb = background
        mass_type = "enhancement_above_custom_background"

    else:
        raise ValueError(
            "background must be 'quantile', None, a number, or an xarray.DataArray."
        )

    # Remove negative enhancements if requested
    if clip_negative:
        delta_ppb = delta_ppb.where(delta_ppb > 0, 0)

    # -----------------------------
    # Conversion
    # -----------------------------
    dry_air_column_kg_m2 = surface_pressure / G
    dry_air_column_mol_m2 = dry_air_column_kg_m2 / M_AIR

    ch4_mol_m2 = delta_ppb * 1e-9 * dry_air_column_mol_m2
    ch4_kg_m2 = ch4_mol_m2 * M_CH4
    ch4_g_m2 = ch4_kg_m2 * 1000.0
    ch4_tonnes_km2 = ch4_kg_m2 * 1000.0

    tonnes_per_pixel = (ch4_kg_m2 * area_m2) / 1000.0

    total_tonnes = tonnes_per_pixel.sum(
        dim=spatial_dims,
        skipna=True
    )

    total_kg = total_tonnes * 1000.0

    valid_area_m2 = area_m2.where(ch4_kg_m2.notnull()).sum(
        dim=spatial_dims,
        skipna=True
    )

    mean_kg_m2 = total_kg / valid_area_m2

    total_mol = (ch4_mol_m2 * area_m2).sum(
        dim=spatial_dims,
        skipna=True
    )

    # -----------------------------
    # Metadata
    # -----------------------------
    attrs = {
        "background_ppb": ("ppb", "Background methane concentration"),
        "delta_ppb": ("ppb", "Methane enhancement above background"),
        "ch4_mol_m2": ("mol m-2", "Methane column amount"),
        "ch4_kg_m2": ("kg m-2", "Methane mass per unit area"),
        "ch4_g_m2": ("g m-2", "Methane mass per unit area"),
        "ch4_tonnes_km2": ("tonnes km-2", "Methane mass per unit area"),
        "tonnes_per_pixel": ("tonnes pixel-1", "Methane mass per pixel"),
        "total_tonnes": ("tonnes", "Total methane mass over valid pixels"),
        "total_kg": ("kg", "Total methane mass over valid pixels"),
        "mean_kg_m2": ("kg m-2", "Mean methane mass per unit area"),
        "valid_area_m2": ("m2", "Valid observation area"),
        "total_mol": ("mol", "Total methane amount over valid pixels"),
    }

    outputs = {
        "background_ppb": background_ppb,
        "delta_ppb": delta_ppb,
        "ch4_mol_m2": ch4_mol_m2,
        "ch4_kg_m2": ch4_kg_m2,
        "ch4_g_m2": ch4_g_m2,
        "ch4_tonnes_km2": ch4_tonnes_km2,
        "tonnes_per_pixel": tonnes_per_pixel,
        "total_tonnes": total_tonnes,
        "total_kg": total_kg,
        "mean_kg_m2": mean_kg_m2,
        "valid_area_m2": valid_area_m2,
        "total_mol": total_mol,
    }

    for name, da in outputs.items():
        da.attrs["units"] = attrs[name][0]
        da.attrs["long_name"] = attrs[name][1]

    ds_out = xr.Dataset(outputs)

    ds_out.attrs["mass_type"] = mass_type
    ds_out.attrs["background"] = str(background)
    ds_out.attrs["background_quantile"] = background_quantile
    ds_out.attrs["note"] = (
        "For emission studies, use methane enhancement above background, "
        "not total atmospheric XCH4."
    )

    return ds_out

def plot_density(
    ch4_ppb,
    quantile=0.05,
    time_dim="time",
    time_index=0,
    max_points=200_000,
    bins=50,
    title="Methane ppb Density Plot and 5th Percentile Background"
):
    """
    Plot a smooth methane ppb density curve and mark the selected quantile.

    The vertical line shows the selected quantile value in ppb.
    The point shows where the quantile meets the density curve.

    Parameters
    ----------
    ch4_ppb : xarray.DataArray
        Sentinel-5P methane XCH4 in ppb.

    quantile : float
        Quantile to identify.
        Example: 0.05 means the 5th percentile.

    time_dim : str
        Name of the time dimension.

    time_index : int
        Timestep to plot if time dimension exists.

    max_points : int
        Maximum number of valid points used for plotting.

    bins : int
        Number of histogram bins.

    title : str
        Plot title.

    Returns
    -------
    float
        Quantile value in ppb.
    """

    ch4_ppb = rename_latlon(ch4_ppb)

    if not isinstance(ch4_ppb, xr.DataArray):
        raise TypeError(
            "ch4_ppb must be an xarray.DataArray. "
            "Select the methane variable first, for example: "
            "ds['methane_mixing_ratio']."
        )

    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1.")

    # Select one timestep
    if time_dim in ch4_ppb.dims:
        da = ch4_ppb.isel({time_dim: time_index})
        plot_time = str(ch4_ppb[time_dim].values[time_index])[:19]
    else:
        da = ch4_ppb
        plot_time = ""

    # Extract valid values
    values = da.values.ravel()
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("No valid CH4 ppb values found.")

    if np.unique(values).size < 2:
        raise ValueError("Density plot needs at least two unique CH4 values.")

    # Sample if dataset is very large
    if values.size > max_points:
        rng = np.random.default_rng(42)
        values = rng.choice(values, size=max_points, replace=False)

    # Calculate quantile
    q_value = np.nanquantile(values, quantile)

    # Smooth density curve
    kde = gaussian_kde(values)
    x_grid = np.linspace(values.min(), values.max(), 500)
    density = kde(x_grid)

    # Density value at selected quantile
    q_density = float(kde([q_value])[0])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        values,
        bins=bins,
        density=True,
        alpha=0.25,
        label="CH4 ppb distribution"
    )

    ax.plot(
        x_grid,
        density,
        linewidth=2.5,
        label="Smooth density curve"
    )

    ax.vlines(
        q_value,
        ymin=0,
        ymax=q_density,
        linestyles="--",
        linewidth=2,
        label=f"{int(quantile * 100)}th percentile = {q_value:.2f} ppb"
    )

    ax.hlines(
        q_density,
        xmin=x_grid.min(),
        xmax=q_value,
        linestyles="--",
        linewidth=1.8
    )

    ax.scatter(
        q_value,
        q_density,
        s=90,
        zorder=5
    )

    ax.annotate(
        f"{int(quantile * 100)}th percentile\n{q_value:.2f} ppb",
        xy=(q_value, q_density),
        xytext=(q_value, q_density * 1.2),
        arrowprops=dict(arrowstyle="->"),
        ha="center",
        fontsize=11
    )

    ax.set_title(f"{title}\n{plot_time}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Methane concentration (ppb)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()

    print(f"{int(quantile * 100)}th percentile methane background: {q_value:.2f} ppb")

    return q_value

def estimate_ch4_emission_rate(
    total_tonnes,
    wind_speed_m_s,
    plume_length_m
):
    """
    Estimate methane emission rate using a simple Integrated Mass Enhancement approach.

    Q = wind speed × total plume mass / plume length

    Parameters
    ----------
    total_tonnes : xarray.DataArray or float
        Total excess methane mass in tonnes.

    wind_speed_m_s : float
        Effective wind speed in m/s.

    plume_length_m : float
        Plume length in metres.

    Returns
    -------
    xarray.Dataset
        Emission rates in kg/s, kg/h, tonnes/h, tonnes/day, and kt/year.
    """

    if plume_length_m <= 0:
        raise ValueError("plume_length_m must be greater than zero.")

    if wind_speed_m_s < 0:
        raise ValueError("wind_speed_m_s cannot be negative.")

    total_kg = total_tonnes * 1000.0

    q_kg_s = (wind_speed_m_s * total_kg) / plume_length_m
    q_kg_h = q_kg_s * 3600.0
    q_tonnes_h = q_kg_h / 1000.0
    q_tonnes_day = q_tonnes_h * 24.0
    q_kt_year = (q_tonnes_day * 365.0) / 1000.0

    ds_emission = xr.Dataset(
        {
            "emission_kg_s": q_kg_s,
            "emission_kg_h": q_kg_h,
            "emission_tonnes_h": q_tonnes_h,
            "emission_tonnes_day": q_tonnes_day,
            "emission_kt_year": q_kt_year,
        }
    )

    ds_emission["emission_kg_s"].attrs["units"] = "kg s-1"
    ds_emission["emission_kg_h"].attrs["units"] = "kg h-1"
    ds_emission["emission_tonnes_h"].attrs["units"] = "tonnes h-1"
    ds_emission["emission_tonnes_day"].attrs["units"] = "tonnes day-1"
    ds_emission["emission_kt_year"].attrs["units"] = "kt year-1"

    ds_emission.attrs["method"] = "Simple Integrated Mass Enhancement approach"
    ds_emission.attrs["note"] = (
        "This is a simplified emission-rate estimate. "
        "Results depend strongly on wind speed and plume length."
    )

    return ds_emission

def plot_data(
    ch4_mass,
    plot_vars=None,
    time_index=0,
    time_dim="time",
    cmap="RdYlGn_r",
    max_cols=3,
    figsize_per_plot=(6, 5),
    title="Sentinel-5P Methane Conversion Products",
):
    """
    Plot Sentinel-5P methane conversion products using a flexible grid layout.

    The number of rows and columns is automatically calculated
    from the number of variables in plot_vars.

    Parameters
    ----------
    ch4_mass : xarray.Dataset
        Output dataset from the methane conversion function.

    plot_vars : list, optional
        List of variables to plot.

    time_index : int
        Time index to plot if the variables contain a time dimension.

    time_dim : str
        Name of the time dimension.

    cmap : str
        Matplotlib colour map.

    max_cols : int
        Maximum number of columns in the plot layout.

    figsize_per_plot : tuple
        Width and height for each subplot.

    title : str
        Main figure title.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """

    if not isinstance(ch4_mass, xr.Dataset):
        raise TypeError("ch4_mass must be an xarray.Dataset.")

    if plot_vars is None:
        plot_vars = [
            "delta_ppb",
            "ch4_mol_m2",
            "ch4_kg_m2",
            "ch4_g_m2",
            "ch4_tonnes_km2",
            "tonnes_per_pixel",
        ]

    missing_vars = [var for var in plot_vars if var not in ch4_mass]

    if missing_vars:
        raise ValueError(f"These variables are missing from ch4_mass: {missing_vars}")

    n_plots = len(plot_vars)

    # Automatically calculate columns and rows
    ncols = min(max_cols, n_plots)
    nrows = math.ceil(n_plots / ncols)

    figsize = (
        figsize_per_plot[0] * ncols,
        figsize_per_plot[1] * nrows
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        constrained_layout=True
    )

    # Make axes iterable even if there is only one plot
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for ax, var in zip(axes, plot_vars):

        da = ch4_mass[var]

        # Select one timestep if time dimension exists
        if time_dim in da.dims:
            da_plot = da.isel({time_dim: time_index}).squeeze()
            plot_time = str(da[time_dim].values[time_index])[:19]
        else:
            da_plot = da.squeeze()
            plot_time = ""

        units = da.attrs.get("units", "")
        long_name = da.attrs.get("long_name", var)

        da_plot.plot(
            ax=ax,
            cmap=cmap,
            robust=True,
            add_colorbar=True,
            cbar_kwargs={
                "label": f"{long_name} ({units})" if units else long_name
            }
        )

        ax.set_title(f"{long_name}\n{plot_time}", fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Hide empty plots
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold"
    )

    plt.show()

    return fig, axes