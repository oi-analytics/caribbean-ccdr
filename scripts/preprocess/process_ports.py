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
from tqdm import tqdm
tqdm.pandas()
from utils import *

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    countries = ['dma']
    for country in countries:
        areas = gpd.read_file(os.path.join(processed_data_path,
                                "infrastructure/transport",
                                f"{country}_ports.gpkg"), 
                        layer='areas')
        print (areas["geometry"])
        # edges.to_file(os.path.join(processed_data_path,
        #                         "infrastructure/transport",
        #                         f"{country}_roads.gpkg"), 
        #                 layer='edges',driver="GPKG")  

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
