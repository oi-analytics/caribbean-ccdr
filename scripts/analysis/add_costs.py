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
from analysis_utils import *
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
    df[f"{cost_type}_cost_min"] = 0.001*df["lanes"]*df[f"{cost_type}_cost_min"]
    df[f"{cost_type}_cost_max"] = 0.001*df["lanes"]*df[f"{cost_type}_cost_max"]
    df[f"{cost_type}_cost_unit"] = "US$/m"

    return gpd.GeoDataFrame(df,geometry="geometry",crs=road_crs)

def energy_cost_multiplier(x,cost_label,x_layer):
    if x[cost_label] != 0:
        if x["cost_unit"] == "US$/MW" and x_layer =="areas":
            return x[cost_label]*x["modified_capacity"]/x.geometry.area, "US$/sq-m"
        elif x["cost_unit"] == "US$/MW":
            return x[cost_label]*x["modified_capacity"], "US$"
        elif x_layer == "areas":
            return x[cost_label]/x.geometry.area, "US$/sq-m"
        else:
            return x[cost_label],x["cost_unit"]  
    else:
        return x[cost_label],x["cost_unit"]

def add_energy_costs(energy_df,cost_df,energy_layer,cost_type="rehab",geometry_type=True):
    cost_df["asset_type"] = cost_df["asset_type"].str.lower()
    
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
    if geometry_type is True:
        energy_crs = energy_df.crs
        return gpd.GeoDataFrame(energy_df,geometry="geometry",crs=energy_crs)
    else:
        return energy_df

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

def add_installed_capacity_costs(country,sector,subsector,cost_types,time_epoch,scenario):
    processed_data_path = load_config()['paths']['data']
    for cost_type in cost_types:
        if sector == "water":
            sheet_name = sector
        else:
            sheet_name = subsector
        costs = pd.read_excel(os.path.join(
                            processed_data_path,
                            "costs_and_options",
                            f"asset_{cost_type}_costs.xlsx"),
                            sheet_name=sheet_name)

        if subsector == "energy":
            installed_capacities = pd.read_excel(os.path.join(
                            processed_data_path,
                            "data_layers",
                            "service_targets_sdgs.xlsx"),
                            sheet_name="sdg_energy_supply")[["iso_code","asset_type","epoch",f"{scenario}_capacity_mw"]]
            installed_capacities = installed_capacities[
                                (installed_capacities["epoch"] == time_epoch
                                ) & (installed_capacities["iso_code"] == country)
                                ]
            installed_capacities["asset_type"] = installed_capacities["asset_type"].str.lower()
            installed_capacities.rename(columns={f"{scenario}_capacity_mw":"modified_capacity"},inplace=True)
            installed_capacities = add_energy_costs(installed_capacities,costs,"none",cost_type=cost_type,geometry_type=False) 

    return installed_capacities


def add_costs(gdf,country,sector,subsector,layer,cost_types,time_epoch,scenario):
    processed_data_path = load_config()['paths']['data']
    gdf = gdf.to_crs(epsg=CARIBBEAN_CRS)
    for cost_type in cost_types:
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
            # Modify capacity column names for Saint Lucia. Should have done it while creating layer!!!!
            gdf["asset_type"] = gdf["asset_type"].str.lower()
            gdf.rename(columns={"supply_capacity_MW":"capacity_mw","supply_capacity_GWH":"capacity_gwh"},inplace=True)
            if "capacity_mw" in gdf.columns.values.tolist():
                installed_capacities = pd.read_excel(os.path.join(
                                processed_data_path,
                                "data_layers",
                                "service_targets_sdgs.xlsx"),
                                sheet_name="sdg_energy_supply")
                installed_capacities = installed_capacities[
                                    (installed_capacities["epoch"] == time_epoch
                                    ) & (installed_capacities["iso_code"] == country)
                                    ]
                installed_capacities["asset_type"] = installed_capacities["asset_type"].str.lower()
                gdf["total_capacity_mw"] = gdf["capacity_mw"].groupby(gdf["asset_type"]).transform('sum')
                gdf = pd.merge(gdf,
                            installed_capacities[["asset_type",f"{scenario}_capacity_mw"]],
                            how="left",on=["asset_type"]).fillna(0)
                gdf["modified_capacity"] = np.where(gdf["total_capacity_mw"] > 0, 
                                            gdf["capacity_mw"]*gdf[f"{scenario}_capacity_mw"]/gdf["total_capacity_mw"],
                                            0)
                gdf["modified_capacity"] = np.where(
                                            gdf["asset_type"] == 'solar',
                                            gdf["capacity_mw"],gdf["modified_capacity"])
                gdf.drop(["total_capacity_mw",f"{scenario}_capacity_mw"],axis=1,inplace=True)
            gdf = add_energy_costs(gdf,costs,layer,cost_type=cost_type) 
        elif sector == "water":
            gdf = add_water_costs(gdf,costs,cost_type=cost_type) 
        else:
            gdf["asset_type"] = subsector
            gdf = add_area_costs(gdf,costs,cost_type=cost_type)

    return gdf

def get_adaptation_uplifts_reductions():
    processed_data_path = load_config()['paths']['data']
    costs = pd.read_excel(os.path.join(
                        processed_data_path,
                        "costs_and_options",
                        "adaptation_options_costs.xlsx"),
                        sheet_name="options")
    index_columns = ["sector","subsector","hazard_type","asset_name"]
    rebuild_option = costs.groupby(
                            index_columns,dropna=False)["rebuild_asset"].apply(list).reset_index(name="rebuild_asset")
    rebuild_option["rebuild_asset"] = rebuild_option.apply(lambda x:"yes" if "yes" in x["rebuild_asset"] else "no",axis=1)
    adaptation_costs = costs.groupby(
                            index_columns)["cost_uplift"].sum().reset_index()
    costs = costs.set_index(index_columns)
    index_values = list(set(costs.index.values.tolist()))
    reductions = []
    for idx in index_values:
        vals = costs.loc[[idx],"vulnerability_reduction"].values.tolist()
        total = vals[0]
        if len(vals) > 1:
            for v in range(1,len(vals)):
                total += vals[v]*np.prod(1-np.array(vals[:v]))

        reductions.append(tuple(list(tuple(idx)) + [total]))
    reductions = pd.DataFrame(reductions,columns=index_columns + ["vulnerability_reduction"])
    adaptation_costs = pd.merge(adaptation_costs,reductions,how="left",on=index_columns)
    adaptation_costs = pd.merge(adaptation_costs,rebuild_option,how="left",on=index_columns)
    adaptation_costs["subsector"] = adaptation_costs["subsector"].str.lower()
    adaptation_costs["sector"] = adaptation_costs["sector"].str.lower()
    return adaptation_costs

if __name__ == '__main__':
    adaptation_costs = get_adaptation_uplifts_reductions()
    adaptation_costs.to_csv("test.csv")