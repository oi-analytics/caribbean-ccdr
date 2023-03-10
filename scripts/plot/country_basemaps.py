"""Generate hazard-damage curves
"""
import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plot_utils import *
from tqdm import tqdm
tqdm.pandas()
CARIBBEAN_GRID_EPSG = 32620

def main(config):
    processed_data_path = config['paths']['data']
    figures_data_path = config['paths']['figures']
    map_expand_factor = 0.001
    countries = ['DMA','GRD','LCA','VCT']
    for country in countries:
        boundary = gpd.read_file(
                    os.path.join(
                        processed_data_path,
                        "admin_boundaries",
                        f"gadm41_{country}.gpkg"),
                    layer="ADM_ADM_0")
        boundary = boundary.to_crs(epsg=CARIBBEAN_GRID_EPSG)
        regions = gpd.read_file(
                    os.path.join(
                        processed_data_path,
                        "admin_boundaries",
                        f"gadm41_{country}.gpkg"),
                    layer="ADM_ADM_1")
        regions = regions.to_crs(epsg=CARIBBEAN_GRID_EPSG)
        bounds = boundary.geometry.total_bounds # this gives your boundaries of the map as (xmin,ymin,xmax,ymax))
        # convert boundaries of the map as (xmin,xmax,ymin,ymax))
        map_extent = ((1 - map_expand_factor)*bounds[0],
                        (1 + map_expand_factor)*bounds[2],
                        (1 - map_expand_factor)*bounds[1],
                        (1 + map_expand_factor)*bounds[3])
        fig, ax = plt.subplots(1,1,
                            subplot_kw={'projection':get_projection(epsg=CARIBBEAN_GRID_EPSG)},
                            figsize=(12,8),
                            dpi=500)
        ax = get_axes(ax,extent=map_extent)
        plot_basemap(ax, boundary,regions,plot_regions=True, region_labels=True)
        save_fig(os.path.join(figures_data_path, f"{country}_map.png"))


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
