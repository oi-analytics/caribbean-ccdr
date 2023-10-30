"""This script allows us to select and parallelise the Damage and Loss estimations on a server with multiple core processors
"""
import os
import sys
import ujson
import itertools
import geopandas as gpd
import pandas as pd
import numpy as np
from analysis_utils import *
import subprocess 

def main(config):
    processed_data_path = config['paths']['data']
    results_path = config['paths']['results']

    # countries = ["dma","grd","lca","vct"]
    countries = ["lca"]
    hazards = ["charim_landslide","deltares_storm_surge","fathom_pluvial_fluvial","chaz_cyclones"]

    hazard_file = pd.read_csv(os.path.join(processed_data_path,"hazards","fathom_pluvial_fluvial_grd.csv"))
    hazard_keys = hazard_file["key"].values.tolist()
    sectors  = ["airports","ports"]
    for sector in sectors:
        file_name = f"grd_{sector}_splits__fathom_pluvial_fluvial_grd__areas.geoparquet"
        df = gpd.read_parquet(os.path.join(results_path,"hazard_asset_intersections",file_name))
        df[hazard_keys] = np.where(df[hazard_keys] > 990,0,df[hazard_keys])
        print (df)
        df.to_parquet(os.path.join(results_path,"hazard_asset_intersections",file_name),index=False)

    # df = gpd.read_file(os.path.join(processed_data_path,"infrastructure","transport","lca_airports.gpkg"),layer="areas")
    # df["area"] = df.geometry.area
    # print (df[["node_id","area"]])
    # print (949212.16/df["area"].sum())
                                
if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)