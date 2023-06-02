#!/usr/bin/env python
# coding: utf-8
"""Add enrollment numbers to school locations 
"""
import os
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from utils import *

def match_buildings_to_adminarea(buildings,buildings_id_column,admin_areas,admin_column):
    matches = gpd.sjoin(buildings,
                        admin_areas, 
                        how="inner", 
                        predicate='intersects').reset_index()
    # print (matches)
    matches.rename(columns={"geometry":"building_geometry"},inplace=True)
    matches = pd.merge(matches, admin_areas[[admin_column,'geometry']],how="left",on=[admin_column])
    matches["area_match"] = matches.progress_apply(
                            lambda x:x["building_geometry"].intersection(x["geometry"].buffer(0)).area,
                            axis=1)

    matches = matches.sort_values(by=["area_match"],ascending=False)
    matches = matches.drop_duplicates(subset=[buildings_id_column], keep="first")
    matches.drop("geometry",axis=1,inplace=True)
    matches.rename(columns={"building_geometry":"geometry"},inplace=True)

    return matches

def nearest_name(x,nodes,name_column):
    gdf = nodes[~nodes[name_column].isna()]
    highest_match = 0
    for idx,g in gdf.iterrows():
        m = SequenceMatcher(None, x.name, g[name_column]).ratio()
        if m >= highest_match:
            highest_match = m
            nearest_name = g[name_column]
            nearest_id = g["ID"]
            nearest_district = g["District"]
            enrollment = g["Enrollment"]
        if highest_match == 1:
            break
    return nearest_id,nearest_name,nearest_district,enrollment,highest_match

def main(config):
    caribbean_epsg = 32620
    global_epsg = 4326
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    
    countries = ['dma','grd','lca','vct']
    health_capacity = pd.read_excel(os.path.join(processed_data_path,
                            "infrastructure/social",
                            "health_capacity_hospitals.xlsx"),
                            sheet_name="bed_capacity")
    health_capacity["bed_capacity"] = health_capacity["bed_capacity"].fillna(0)
    health_capacity = health_capacity[health_capacity["bed_capacity"]>0]
    for country in countries:
        country_health = gpd.read_file(os.path.join(processed_data_path,
                                "infrastructure/social",
                                "archive",
                        f"{country}_health.gpkg"),
                        layer="areas")
        country_health = country_health.to_crs(epsg=caribbean_epsg)
        country_health["area"] = country_health.geometry.area
        cap = pd.merge(health_capacity[health_capacity["iso_code"] == country][["node_id","bed_capacity"]],
                country_health[["node_id","area"]],
                how="left",on=["node_id"])
        cap["bed_persqm"] = cap["bed_capacity"]/cap["area"]
        country_health["bed_capacity"] = country_health["area"]*cap["bed_persqm"].mean()
        country_health["bed_capacity"] = country_health["bed_capacity"].apply(np.ceil)
        country_health.loc[country_health["node_id"].isin(cap["node_id"].values.tolist()),"bed_capacity"] = cap["bed_capacity"].values
        country_health.to_file(os.path.join(processed_data_path,
                                "infrastructure/social",
                        f"{country}_health.gpkg"),
                        layer="areas",driver="GPKG")

        


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
