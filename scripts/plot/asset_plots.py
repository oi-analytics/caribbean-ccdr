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
from caribbean_sector_plotting_attributes import *
from tqdm import tqdm
tqdm.pandas()
CARIBBEAN_GRID_EPSG = 32620

def main(config):
    processed_data_path = config['paths']['data']
    figures_data_path = config['paths']['figures']
    map_expand_factor = 0.001
    countries = ['DMA','GRD','LCA','VCT']
    sector_attributes = caribbean_sector_attributes()
    for country in countries:
        for sector in sector_attributes:
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

            sector_dfs = get_sector_layer(country.lower(),
                                        sector,
                                        os.path.join(processed_data_path,"infrastructure",sector["sector"]))
            if len(sector_dfs) > 0:
                legend_handles = []
                fig, ax = plt.subplots(1,1,
                                subplot_kw={'projection':get_projection(epsg=CARIBBEAN_GRID_EPSG)},
                                figsize=(12,8),
                                dpi=500)
                ax = get_axes(ax,extent=map_extent)
                plot_basemap(ax, boundary,regions,plot_regions=True, region_labels=True)
                for idx, (df,layer_key) in enumerate(sector_dfs):
                    ax, legend_handles = plot_lines_and_points(ax,legend_handles,sector,sector_dataframe=df,layer_key=layer_key)
                    if sector["sector_label"] in ["Ports","Airports"]:
                        ax = add_labels(ax,df,sector["label"])
                if country == "DMA":
                    legend_loc = "lower left"
                else:
                    legend_loc = "upper left"
                ax.legend(handles=legend_handles,fontsize=8,loc=legend_loc) 
                save_fig(os.path.join(figures_data_path, f"{country}_{sector['sector_label'].lower().replace(' ','_')}_system.png"))


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
