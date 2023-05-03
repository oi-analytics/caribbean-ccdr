"""This script allows us to select and parallelise the Damage and Loss estimations on a server with multiple core processors
"""
import os
import sys
import ujson
import itertools
import geopandas as gpd
import pandas as pd
from analysis_utils import *
import subprocess 

def main(config):
    processed_data_path = config['paths']['data']
    results_path = config['paths']['results']

    # countries = ["dma","grd","lca","vct"]
    countries = ["lca"]
    hazards = ["charim_landslide","deltares_storm_surge","fathom_pluvial_fluvial","chaz_cyclones"]

    # file_name = "lca_roads_splits__charim_landslide_lca__edges.geoparquet"
    # df = gpd.read_parquet(os.path.join(results_path,"hazard_asset_intersections",file_name))
    # print (df)
    # df.to_csv("test.csv",index=False)

    df = gpd.read_file(os.path.join(processed_data_path,"infrastructure","transport","lca_airports.gpkg"),layer="areas")
    df["area"] = df.geometry.area
    print (df[["node_id","area"]])
    print (949212.16/df["area"].sum())
                                
if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)