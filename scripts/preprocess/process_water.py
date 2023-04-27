#!/usr/bin/env python
# coding: utf-8
"""Process road data from OSM extracts and create road network topology 
    WILL MODIFY LATER
"""
import os
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
from pyproj import Geod
import networkx
import igraph
import snkit
from shapely.ops import linemerge
from shapely.geometry import Point,LineString,MultiLineString
from tqdm import tqdm
tqdm.pandas()
from utils import *

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    caribbean_crs = 32620

    for asset_type in ["wtp","wwtp"]: 
        lca_df = gpd.read_file(os.path.join(incoming_data_path,
                                "water_data",
                                f"lca_{asset_type}.gpkg"),layer="nodes")

        lca_df = lca_df.to_crs(epsg=caribbean_crs)
        lca_df.rename(columns={"Capacity":"capacity_m3d","Access Value":"capacity_m3d"},inplace=True)
        lca_df["asset_type"] = asset_type
        lca_df["capacity_m3d"] = lca_df["capacity_m3d"]/365.0

        lca_df.to_file(os.path.join(processed_data_path,
                                        "infrastructure/water",
                                        f"lca_{asset_type}.gpkg"),
                                        layer="nodes",
                                        driver="GPKG")

    grd_df = gpd.read_file(os.path.join(incoming_data_path,
                                "water_data",
                                "grd_water.gpkg"))
    grd_df = grd_df.to_crs(epsg=caribbean_crs)
    grd_df["asset_type"] = "wtp"
    grd_df["capacity_m3d"] = grd_df["capacity_mgd"]*3785.0
    grd_df.to_file(os.path.join(processed_data_path,
                                        "infrastructure/water",
                                        "grd_wtp.gpkg"),
                                        layer="nodes",
                                        driver="GPKG")


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
