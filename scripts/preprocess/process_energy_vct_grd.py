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

    for country in ["grd","vct"]:
        plants = gpd.read_file(os.path.join(incoming_data_path,"grd_vct_power",
                            f"{country}_energy.gpkg"))
        plants.rename(columns={"type":"asset_type"},inplace=True)
        plants = plants.to_crs(epsg=32620)
        plants["node_id"] = plants.index.values.tolist()
        plants["node_id"] = plants.progress_apply(lambda x:f"elecn_{x.node_id}",axis=1)
        print (plants)
        gpd.GeoDataFrame(plants,
            geometry="geometry",
            crs="EPSG:32620").to_file(os.path.join(processed_data_path,
                                    "infrastructure/energy",
                                    f"{country}_energy.gpkg"),
                                    layer="nodes",
                                    driver="GPKG")

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
