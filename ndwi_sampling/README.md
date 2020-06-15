Scripts used to generate Landsat 8 all time NDWI median and create stratified random sampling points for each AEZ.

Dependency
* Simplified AEZ shapes
* 150 km x 150 kM tiles covering African continent

Following notebooks:
* NDWI_composite.ipynb: generate NDWI median in tiles.
* Check_ndwi_composite.ipynb: check NDWI tiles don't have excessive nan values in them. 
* Sampling.ipynb: create stratified random sampling from NDWI median.
* WOfS_summary_distribution.ipynb: load wofs summary and check distribution of water detection frequency for each AEZ.
