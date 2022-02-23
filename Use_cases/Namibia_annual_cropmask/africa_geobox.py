from typing import Tuple

from datacube.utils.geometry import GeoBox
from odc.dscache.tools.tiling import GRIDS


class AfricaGeobox:
    """
    generate the geobox for each tile according to the longitude ande latitude bounds.
    add origin to remove the negative coordinate
    x_new = x_old  + 181
    y_new = y_old + 77
    """

    def __init__(self, resolution: int = 10):
        self.albers_africa_N = GRIDS[f"africa_{resolution}"]

    def __getitem__(self, tile_index: Tuple[int, int]) -> GeoBox:
        return self.albers_africa_N.tile_geobox(tile_index)