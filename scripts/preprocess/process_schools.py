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
            enrollment = g["Enrollment"]
        if highest_match == 1:
            break
    return nearest_id,nearest_name,nearest_district,enrollment,highest_match

def main(config):
    caribbean_epsg = 32620
    global_epsg = 4326
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    match_threshold = 0.78
    schools = gpd.read_file(os.path.join(processed_data_path,
                        "infrastructure/social",
                        "education.gpkg"),
                    layer="areas")
    schools = schools.to_crs(epsg=caribbean_epsg)
    schools["area"] = schools.geometry.area
    
    countries = ['DMA','GRD','LCA','VCT']
    all_schools = []
    all_districts = []
    for country in countries:
        country_schools = schools[schools['iso_code'] == country]
        # print (country_schools)
        if country == "LCA":
            admin_areas = gpd.read_file(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "social",
                                "lca_edu_districts.gpkg"),
                            layer="areas")[["DIS","geometry"]]
            admin_areas["DIS"] = admin_areas.progress_apply(lambda x: str(num2words(x["DIS"])).title(),axis=1)
            admin_areas.rename(columns={"DIS":"school_district"},inplace=True)
        elif country == "VCT":
            admin_areas = gpd.read_file(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "social",
                                "schools_districts_VCT.gpkg"))[["SCHOOL_DIST","geometry"]]
            admin_areas["SCHOOL_DIST"] = admin_areas.progress_apply(lambda x: str(num2words(x["SCHOOL_DIST"])).title(),axis=1)
            admin_areas.rename(columns={"SCHOOL_DIST":"school_district"},inplace=True)
        else:
            admin_areas = gpd.read_file(os.path.join(processed_data_path,
                                "admin_boundaries",
                                f"gadm41_{country}.gpkg"),layer="ADM_ADM_1")[["NAME_1","geometry"]]
            admin_areas["NAME_1"] = admin_areas["NAME_1"].str.replace("Carriacou","Carriacou & Petite Martinique")
            admin_areas.rename(columns={"NAME_1":"school_district"},inplace=True)
        admin_areas = admin_areas.to_crs(epsg=caribbean_epsg)
        admin_areas["school_district"] = admin_areas["school_district"].astype(str).str.replace("Saint","St.")
        country_schools = match_buildings_to_adminarea(country_schools,"node_id",admin_areas,"school_district")
        # print (country_schools)

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
        district_data = country_data.groupby(["District"])["Enrollment"].sum().reset_index()
        district_data["iso_code"] = country
        school_year = country_data["Year"].values[0]
        all_districts.append(district_data.copy())
        country_schools["year"] = school_year
        if country != "DMA":
            school_matches = []
            for st in country_schools.itertuples():
                nearest_id, nearest_school, nearest_district, enrollment, match = nearest_name(st,country_data,'School Name')
                if match > match_threshold and st.school_district == nearest_district:
                    school_matches.append((st.node_id,st.name,st.school_district,st.area,
                            nearest_id,nearest_school,nearest_district,enrollment,match)) 
                # school_matches.append((st.node_id,st.name,st.school_district,nearest_id,nearest_school,nearest_district,match))    
            school_matches = pd.DataFrame(school_matches,
                                columns=["node_id","name",
                                        "school_district","area","ID",
                                        "School Name","District",
                                        "Enrollment","Match"])
            school_matches["total_areas"] = school_matches.groupby(["ID","School Name"])["area"].transform('sum')
            school_matches["assigned_students"] = school_matches["Enrollment"]*school_matches["area"]/school_matches["total_areas"]
            school_matches["assigned_students"] = school_matches["assigned_students"].round(0)
            district_matches = school_matches.groupby(["District"])["assigned_students"].sum().reset_index()
            remaining_schools = country_schools[~(country_schools["node_id"].isin(school_matches["node_id"].values.tolist()))]
            country_schools = pd.merge(country_schools,school_matches[["node_id","assigned_students"]],how="left",on=["node_id"])
            all_schools.append(country_schools[country_schools["node_id"].isin(school_matches["node_id"]).values.tolist()])

            district_data = district_data.set_index("District")
            district_matches = district_matches.set_index("District")
            district_data["Enrollment"] = district_data["Enrollment"].sub(
                                                                district_matches["assigned_students"],
                                                                axis='index',fill_value=0
                                                                )
            district_data = district_data.reset_index()
        else:
            remaining_schools = country_schools.copy()

        remaining_schools = pd.merge(remaining_schools,district_data,
                                how="left",left_on=["school_district"],
                                right_on=["District"]).fillna(0)
        remaining_schools["total_areas"] = remaining_schools.groupby(["District"])["area"].transform('sum')
        remaining_schools["assigned_students"] = remaining_schools["Enrollment"]*remaining_schools["area"]/remaining_schools["total_areas"]
        remaining_schools["assigned_students"] = remaining_schools["assigned_students"].round(0)
        remaining_schools["assigned_students"][remaining_schools["assigned_students"] < 0] = 0
        # remaining_schools.drop(["District","Enrollment"],axis=1,inplace=True)
        all_schools.append(remaining_schools)

    all_schools = pd.concat(all_schools,axis=0,ignore_index=True)
    schools = pd.merge(schools,all_schools[["node_id","school_district","assigned_students","year"]],how="left",on=["node_id"])

    # check the totals at the district level
    district_totals = schools.groupby(["iso_code","school_district"])["assigned_students"].sum().reset_index()
    all_districts =  pd.concat(all_districts,axis=0,ignore_index=True)
    district_totals = pd.merge(district_totals,
                        all_districts,how="left",
                        left_on=["iso_code","school_district"],right_on=["iso_code","District"])
    
    schools = gpd.GeoDataFrame(
                        schools,
                        geometry="geometry",
                        crs=f"EPSG:{caribbean_epsg}")
    schools.to_file(os.path.join(
                    processed_data_path,
                    "infrastructure/social",
                    "education.gpkg"),
                    layer="areas",driver="GPKG")


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
