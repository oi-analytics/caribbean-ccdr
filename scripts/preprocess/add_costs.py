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
import fiona
from tqdm import tqdm
tqdm.pandas()
from utils import *
CARIBBEAN_CRS = 32620

def add_road_costs(road_df,cost_df,cost_type="rehab"):
    road_crs = road_df.crs
    bridges = road_df[road_df["bridge"]=="yes"]
    bridge_costs = cost_df[cost_df["bridge"]=="yes"]
    bridges[f"{cost_type}_cost_min"] = bridge_costs["cost_min"].values[0]
    bridges[f"{cost_type}_cost_max"] = bridge_costs["cost_max"].values[0]

    road_df = road_df[road_df["bridge"]!="yes"]
    materials = list(set(cost_df["material"].values.tolist()))
    highways = list(set(cost_df["highway"].values.tolist()))
    hm_df = road_df[(road_df["material"].isin(materials)) & (road_df["highway"].isin(highways))]
    hm_df = pd.merge(hm_df,cost_df[["highway","material","cost_min","cost_max"]],
                    how="left",on=["highway","material"])
    hm_df.rename(columns={"cost_min":f"{cost_type}_cost_min","cost_max":f"{cost_type}_cost_max"},inplace=True)
    
    road_df = road_df[~road_df["edge_id"].isin(hm_df["edge_id"].values.tolist())]
    m_df = road_df[(road_df["material"].isin(materials))]
    m_df = pd.merge(m_df,cost_df[cost_df["highway"]=="other"][["material","cost_min","cost_max"]],
                    how="left",on=["material"])
    m_df.rename(columns={"cost_min":f"{cost_type}_cost_min","cost_max":f"{cost_type}_cost_max"},inplace=True)
    
    road_df = road_df[~road_df["edge_id"].isin(m_df["edge_id"].values.tolist())]
    h_df = road_df[(road_df["highway"].isin(highways))]
    h_df = pd.merge(h_df,cost_df[cost_df["material"]=="other"][["highway","cost_min","cost_max"]],
                    how="left",on=["highway"])
    h_df.rename(columns={"cost_min":f"{cost_type}_cost_min","cost_max":f"{cost_type}_cost_max"},inplace=True)
    
    road_df = road_df[~road_df["edge_id"].isin(h_df["edge_id"].values.tolist())]
    r_costs = cost_df[(cost_df["highway"]=="other") & (cost_df["material"]=="other")]
    road_df[f"{cost_type}_cost_min"] = r_costs["cost_min"].values[0]
    road_df[f"{cost_type}_cost_max"] = r_costs["cost_max"].values[0]

    df = pd.concat(
                    [road_df,bridges,hm_df,m_df,h_df],
                    axis=0,
                    ignore_index=True
                )
    df[f"{cost_type}_cost_min"] = df["lanes"]*df[f"{cost_type}_cost_min"]
    df[f"{cost_type}_cost_max"] = df["lanes"]*df[f"{cost_type}_cost_max"]
    df[f"{cost_type}_cost_unit"] = "US$/km"

    return gpd.GeoDataFrame(df,geometry="geometry",crs=road_crs)

def energy_cost_multiplier(x,cost_label,x_layer):
    if x[cost_label] != 0:
        if x["cost_unit"] == "US$/MW" and x_layer =="areas":
            return x[cost_label]*x["capacity_mw"]/x.geometry.area, "US$/sq-m"
        elif x["cost_unit"] == "US$/MW":
            return x[cost_label]*x["capacity_mw"], "US$"
        elif x_layer == "areas":
            return x[cost_label]/x.geometry.area, "US$/sq-m"
        else:
            return x[cost_label],x["cost_unit"]  
    else:
        return x[cost_label],x["cost_unit"]

def add_energy_costs(energy_df,cost_df,energy_layer,cost_type="rehab"):
    energy_crs = energy_df.crs
    energy_df["asset_type"] = energy_df["asset_type"].str.lower()
    cost_df["asset_type"] = cost_df["asset_type"].str.lower()

    # Modify capacity column names for Saint Lucia. Should have done it while creating layer!!!!
    energy_df.rename(columns={"supply_capacity_MW":"capacity_mw","supply_capacity_GWH":"capacity_gwh"},inplace=True)
    
    energy_df = pd.merge(energy_df,cost_df,
                    how="left",on=["asset_type"]).fillna(0)
    energy_df.rename(columns={"cost_min":f"{cost_type}_cost_min","cost_max":f"{cost_type}_cost_max"},inplace=True)
    energy_df[f"{cost_type}_costs"] = energy_df.progress_apply(
                                            lambda x:energy_cost_multiplier(x,f"{cost_type}_cost_min",energy_layer),
                                            axis=1)
    energy_df[[f"{cost_type}_cost_min",f"{cost_type}_cost_unit"]] = energy_df[f"{cost_type}_costs"].apply(pd.Series)

    energy_df[f"{cost_type}_costs"] = energy_df.progress_apply(
                                            lambda x:energy_cost_multiplier(x,f"{cost_type}_cost_max",energy_layer),
                                            axis=1)
    energy_df[[f"{cost_type}_cost_max",f"{cost_type}_cost_unit"]] = energy_df[f"{cost_type}_costs"].apply(pd.Series)
    energy_df.drop(["cost_unit",f"{cost_type}_costs"],axis=1,inplace=True)

    return gpd.GeoDataFrame(energy_df,geometry="geometry",crs=energy_crs)

def add_water_costs(water_df,cost_df,cost_type="rehab"):
    water_crs = water_df.crs
    water_df = pd.merge(water_df,cost_df,how="left",on=["asset_type"])
    water_df.drop("cost_unit",axis=1,inplace=True)
    water_df.rename(columns={"cost_min":f"{cost_type}_cost_min","cost_max":f"{cost_type}_cost_max"},inplace=True)
    water_df[f"{cost_type}_cost_min"] = water_df[f"{cost_type}_cost_min"]*water_df["capacity_m3d"]
    water_df[f"{cost_type}_cost_max"] = water_df[f"{cost_type}_cost_max"]*water_df["capacity_m3d"]
    water_df[f"{cost_type}_cost_unit"] = "US$"
    return water_df

def add_area_costs(area_df,cost_df,cost_type="rehab"):
    area_df[f"{cost_type}_cost_min"] = cost_df["cost_min"].values[0]
    area_df[f"{cost_type}_cost_max"] = cost_df["cost_max"].values[0]
    area_df[f"{cost_type}_cost_unit"] = "US$/sq-m"

    return area_df

def main(config):
    processed_data_path = config['paths']['data']

    countries = ["dma","grd","lca","vct"]
    # countries = ["dma"]
    sectors = {
                "transport":[("roads","edges"),("ports","areas"),("airports","areas")],
                "energy":[("energy","nodes"),("energy","edges"),("energy","areas")],
                "water":[("wtp","nodes"),("wwtp","nodes")],
                "social":[("education","areas"),("health","areas")]
            }
    for country in countries:
        for sector,subsectors in sectors.items():
            for idx,(subsector,layer) in enumerate(subsectors):
                read_gpkg = os.path.join(
                                    processed_data_path,
                                    "infrastructure",
                                    sector,
                                    f"{country}_{subsector}.gpkg")
                if os.path.isfile(read_gpkg):
                    layers = fiona.listlayers(read_gpkg)
                    if layer in layers:
                        gdf = gpd.read_file(read_gpkg,layer=layer)
                        gdf = gdf.to_crs(epsg=CARIBBEAN_CRS)
                        for cost_type in ["rehabilitation","construction"]:
                            if sector == "water":
                                sheet_name = sector
                            else:
                                sheet_name = subsector
                            costs = pd.read_excel(os.path.join(
                                                processed_data_path,
                                                "costs_and_options",
                                                f"asset_{cost_type}_costs.xlsx"),
                                                sheet_name=sheet_name)
                            for c in ["cost_min","cost_max","cost_unit"]:
                                if f"{cost_type}_{c}" in gdf.columns.values.tolist():
                                    gdf.drop(f"{cost_type}_{c}",axis=1,inplace=True)

                            if subsector == "roads":
                                gdf = add_road_costs(gdf,costs,cost_type=cost_type)
                            elif subsector == "energy":
                                gdf = add_energy_costs(gdf,costs,layer,cost_type=cost_type) 
                            elif sector == "water":
                                gdf = add_water_costs(gdf,costs,cost_type=cost_type) 
                            else:
                                gdf["asset_type"] = subsector
                                gdf = add_area_costs(gdf,costs,cost_type=cost_type)

                        gdf.to_file(os.path.join(
                                    processed_data_path,
                                    "infrastructure",
                                    sector,
                                    f"{country}_{subsector}.gpkg"),layer=layer,driver="GPKG")



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
