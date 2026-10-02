# Kenya insect diversity & intactness – results

Results from the final run of [`Biodiversity_Code (2).ipynb`](Biodiversity_Code%20(2).ipynb), run on the
DE Africa Sandbox. Outputs are written to `Data/Outputs_2/`.

## Data

| Input | Details |
|---|---|
| Occurrence points | `Data/Shapefiles/Kenya_presence_points.csv`: **541** presence points, all inside the Kenya AOI |
| Human impact (HFI) | `HumanImpact` column of the occurrence CSV, range **28.3 – 2877.6** |
| Potential habitat (P) | `Data/Kenya/potential-un.tif`: UN Biodiversity Portal (IUCN habitat) class codes, EPSG:4326, ~250 m (0.00225°), uint16 |
| Background points | 541 sampled per scenario; 80/20 train/test split (train 664 = 432 presence + 232 background; test 168 = 109 + 59) |

## 1. Species distribution models (sections 4–5)

Random Forest presence/background model. The table shows performance on the held-out test set; the threshold is Youden's J.

| Scenario | Predictors | AUC | Sensitivity | Specificity | Accuracy | Precision | F1 |
|---|---|---|---|---|---|---|---|
| GeoMAD | 13 (10 S2 bands + EMAD, SMAD, BCMAD) | 0.713 | **0.881** | 0.458 | **0.732** | 0.750 | **0.810** |
| GeoMAD_Climate_Canopy_HF | – | *not run* | | | | | |
| TCI | 3 (greenness, brightness, wetness) | 0.679 | 0.661 | **0.627** | 0.649 | 0.766 | 0.709 |
| TCI_Climate_Canopy_HF | 7 (TCI + canopy height, population impact, temperature, precipitation) | **0.741** | 0.734 | 0.610 | 0.691 | **0.777** | 0.755 |

- **Best discrimination:** TCI_Climate_Canopy_HF (AUC 0.741). Adding climate, canopy height and population impact
  to the three Tasseled Cap indices raised AUC by 0.06.
- **GeoMAD:** it has the highest sensitivity, accuracy and F1, but the lowest specificity (0.458). It over-predicts
  suitable habitat at background points.
- **VIF:** no predictor was removed in any scenario (threshold 2000), and no layer was empty.
- **GeoMAD_Climate_Canopy_HF was skipped:** `Data/Kenya/Kenya_Geomad_SPECTRAL_2020_2023_250M` has no rasters on the Sandbox.
  The rasters exist locally in `Kenya_Geomad_SPECTRAL_2020_2023_250M.zip` (Sentinel-2 bands, MADs, CanopyHeight,
  HumanImpact, precipitation, temperature) but were not uploaded.

The per-scenario figures (ROC, variable importance, response curves, diversity maps) and
`Scenario_Performance_Summary.csv` / `Fig_Cross_Scenario_Comparison.png` are in `Data/Outputs_2/`.

## 2. Intactness index (section 6)

**Pre-weighted Anthropogenic Degradation**

> Intactness = (D × (1 − HFI)) / P

- **D**: current diversity, the RF-predicted probability from the scenario's `<scenario>_Insect_Diversity.tif`.
- **HFI**: the CSV `HumanImpact`, min-max scaled to 0–1 over the 541 occurrence points.
- **P**: potential habitat. The raster class codes are reclassified as follows:

| Raster code | Ecosystem | P | Occurrence points |
|---|---|---|---|
| 100 | Forest | 1.0 | 214 |
| 900–1300 | Aquatic / Marine | 0.9 | 1 |
| 200 | Savanna | 0.7 | 215 |
| 300 | Shrubland | 0.6 | 87 |
| 400 | Grassland | 0.5 | 24 |
| 500 | Wetlands | 0.4 | 0 |
| 600 | Rocky Areas | 0.25 | 0 |
| 800 | Desert | 0.1 | 0 |

Every occurrence point falls in a reclassified class, so none lack a P value.

### Results at occurrence points

| Scenario | Median intactness | % in [0, 1] | Spearman ρ vs HFI | Tree Cover mean | Cropland mean | Range check (≥ 90 % in [0, 1]) | Decreases with HFI | Tree > Crop |
|---|---|---|---|---|---|---|---|---|
| GeoMAD | 0.569 | 93.7 | −0.875 | 0.616 | 0.387 | ✅ | ✅ | ✅ |
| TCI | 0.627 | 92.1 | −0.841 | 0.636 | 0.446 | ✅ | ✅ | ✅ |
| TCI_Climate_Canopy_HF | 0.646 | 89.6 | −0.851 | 0.643 | 0.450 | ❌ (just under 90 %) | ✅ | ✅ |

The Spearman correlations are all significant (p ≈ 0).

**Interpretation**
- **Declines with human pressure:** in all three scenarios, intactness falls strongly and consistently as human pressure rises (ρ −0.84 to −0.88).
  At the highest HFI values it approaches 0, as expected.
- **Land-cover ordering:** Tree Cover points average far above Cropland points (0.62–0.64 vs 0.39–0.45).
  Cropland also has the lowest median of all land covers.
- **Values mostly in 0–1:** 90–94 % of points fall in [0, 1], so the index largely behaves as a proportion of potential remaining.
  The maximum is ~1.7–1.8.
- **Values above 1:** these occur where predicted diversity D exceeds the potential score P, and are mostly low-HFI points in the
  lower-P classes (Shrubland 0.6, Grassland 0.5). Dividing by a small P inflates the index there. As a result, pixels in the
  P = 0.5 and P = 0.6 classes have a *higher* median intactness (~0.63–0.69) than Forest pixels (P = 1.0, ~0.54)
  (GeoMAD, section 6c figure).
- **Scenario differences:** the scenarios rank the points almost identically. The differences come from D: GeoMAD
  gives the lowest values, TCI_Climate_Canopy_HF the highest.

## 3. Intactness maps (section 6b)

HFI only exists at the occurrence points, so section 6b draws the point intactness over the D and P rasters.
It does not produce a full intactness raster.

| Scenario | Points mapped | Mean | Median |
|---|---|---|---|
| GeoMAD | 539 | 0.571 | 0.569 |
| TCI | 541 | 0.617 | 0.627 |
| TCI_Climate_Canopy_HF | 539 | 0.633 | 0.646 |

Two points have no D value under GeoMAD and TCI_Climate_Canopy_HF because they fall on masked (NoData) diversity pixels.

- **D maps:** the highest diversity is in the south-west and central highlands, and the lowest in the arid north and east.
- **P map:** Forest (1.0) dominates the western and central highlands and Savanna (0.7) covers most of the rest of the country.
- **Points:** the occurrence points are concentrated in the south-west, the central highlands and the south-east coast.
  Low-intactness points (red) cluster around the Lake Victoria basin and the central highlands, where human impact is highest.

Each scenario also gets a GeoPackage of the point results (`<scenario>_Intactness_points.gpkg`).

## 4. Intactness on the 250 m grid (section 6c)

Each point's HFI is burned into the 250 m pixel of the diversity map it falls in, and intactness is recomputed per pixel.

| Scenario | Occupied pixels | Valid pixels | Min | Median | Max | % in [0, 1] | Spearman ρ vs HFI |
|---|---|---|---|---|---|---|---|
| GeoMAD | 541 | 539 | 0.0 | 0.580 | 1.706 | 93.7 | −0.876 |
| TCI | 541 | 541 | 0.0 | 0.627 | 1.820 | 92.1 | −0.841 |
| TCI_Climate_Canopy_HF | 541 | 539 | 0.0 | 0.646 | 1.773 | 89.6 | −0.851 |

- **One point per pixel:** no two points share a 250 m pixel, so no averaging was needed.
- **TCI scenarios:** the results match section 6 exactly, because their diversity grids line up with the potential habitat raster.
- **GeoMAD:** the median is slightly higher (0.580 vs 0.569). Its diversity map was reprojected from EPSG:6933, so its
  pixels don't line up exactly with the P raster, and a few points near habitat-class borders pick up a different P value.
- **Outputs:** `Intactness_Exploration/<scenario>/grid_250m/` holds the `*_Intactness_250m.tif`,
  `*_HFI_points_250m.tif` and `*_N_points_250m.tif` rasters, plus `pixel_intactness_250m.csv`.

## Caveats

1. **HFI may be counted twice in TCI_Climate_Canopy_HF.** That model already uses `population_impact` as a predictor, so
   D already reflects human pressure before it is multiplied by (1 − HFI). This probably also explains why it has the most
   values outside [0, 1].
2. **HFI is scaled against the points, not the country.** It is min-max scaled over the occurrence points, so the
   least-impacted point gets HFI = 0 and the most-impacted gets 1. The values are relative to this sample, not to all of Kenya.
3. **Low-P classes inflate the index.** The drivers are identified under section 2; capping at 1 or reporting D × (1 − HFI)
   alongside the index would make the classes easier to compare.
4. **No wall-to-wall intactness map.** HFI is only known at the 541 points; section 6c places these points on the 250 m
   grid but does not fill the gaps between them.
5. **One scenario is missing.** GeoMAD_Climate_Canopy_HF did not run; upload its raster folder to the Sandbox to complete the comparison.
