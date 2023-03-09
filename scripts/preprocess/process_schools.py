#!/usr/bin/env python
# coding: utf-8
"""Add enrollment numbers to school locations 
"""
import os
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
from num2words import num2words
from difflib import SequenceMatcher
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
        if highest_match == 1:
            break
    return nearest_id,nearest_name,nearest_district,highest_match

def main(config):
    caribbean_epsg = 32620
    global_epsg = 4326
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    schools = gpd.read_file(os.path.join(processed_data_path,
                        "infrastructure/social",
                        "education.gpkg"),
                    layer="areas")
    schools = schools.to_crs(epsg=caribbean_epsg)
    schools["area"] = schools.geometry.area
    
    countries = ['GRD','LCA']
    school_matches = []
    for country in countries:
        country_schools = schools[schools['iso_code'] == country]
        print (country_schools)
        if country == "LCA":
            admin_areas = gpd.read_file(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "social",
                                "lca_edu_districts.gpkg"),
                            layer="areas")[["DIS","geometry"]]
            admin_areas["DIS"] = admin_areas.progress_apply(lambda x: str(num2words(x["DIS"])).title(),axis=1)
            admin_areas.rename(columns={"DIS":"school_district"},inplace=True)
        else:
            admin_areas = gpd.read_file(os.path.join(processed_data_path,
                                "admin_boundaries",
                                f"gadm41_{country}.gpkg"),layer="ADM_ADM_1")[["NAME_1","geometry"]]
            admin_areas.rename(columns={"NAME_1":"school_district"},inplace=True)
        admin_areas = admin_areas.to_crs(epsg=caribbean_epsg)
        admin_areas["school_district"] = admin_areas["school_district"].astype(str).str.replace("Saint","St.")
        country_schools = match_buildings_to_adminarea(country_schools,"node_id",admin_areas,"school_district")
        print (country_schools)

        country_schools["name"] = country_schools["name"].astype(str).str.replace(" School","")
        country_schools["name"] = country_schools["name"].astype(str).str.replace(" school","")
        country_schools["name"] = country_schools["name"].astype(str).str.replace(" college","")
        country_schools["name"] = country_schools["name"].astype(str).str.replace(" College","")
        country_data = pd.read_excel(os.path.join(
                                incoming_data_path,
                                "schools",
                                "schools_data.xlsx"),
                            sheet_name=country)
        country_data["School Name"] = country_data["School Name"].astype(str).str.replace(" School","")
        country_data["School Name"] = country_data["School Name"].astype(str).str.replace(" school","")
        country_data["School Name"] = country_data["School Name"].astype(str).str.replace(" college","")
        country_data["School Name"] = country_data["School Name"].astype(str).str.replace(" College","")
        for st in country_schools.itertuples():
            nearest_id, nearest_school, nearest_district,match = nearest_name(st,country_data,'School Name')
            school_matches.append((st.node_id,st.name,st.school_district,nearest_id,nearest_school,nearest_district,match)) 

    school_matches = pd.DataFrame(school_matches,columns=["node_id","name","school_district","ID","School Name","District","Match"])
    school_matches.to_csv(os.path.join(
                        incoming_data_path,
                        "schools",
                        "schools_matches.csv"),index=False)


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
