"""Estimate direct damages to physical assets exposed to hazards

"""
import sys
import os

import pandas as pd
import geopandas as gpd
import numpy as np
import ast
import warnings
import transport_flow_and_disruption_functions as tfdf
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from analysis_utils import *
from tqdm import tqdm
tqdm.pandas()

epsg_caribbean = 32620
TRUNC_THRESH = 0.95
COST = 'time_m'

def get_percentage_exposures_losses(damages_df,sector_total_service,service_disruption_columns):
    for hk in ["min","mean","max"]:
        damages_df[f"exposure_{hk}_percentage"] = 100.0*damages_df[f"exposure_{hk}"]/sector_total_service["asset_area"]
    for sc in service_disruption_columns:
        for hk in ["min","mean","max"]: 
            damages_df[f"{sc}_disrupted_{hk}_percentage"] = 100.0*damages_df[f"{sc}_disrupted_{hk}"]/sector_total_service[f"total_{sc}"]
            damages_df = damages_df.sort_values(by=[f"{sc}_disrupted_{hk}_percentage"],ascending=False)
            damages_df[f"{sc}_disrupted_{hk}_percentage_cumsum"] = damages_df.groupby(
                                                ["hazard","epoch","rcp","rp"],dropna=False
                                                )[f"{sc}_disrupted_{hk}_percentage"].transform(pd.Series.cumsum)

    return damages_df


def add_disruptions(disruptions_df,disruption_columns,hazard_columns):
    sum_dict = dict([
                    (f"exposure_{hk}","sum") for hk in ["min","mean","max"]
                    ]+[
                    (f"damage_{hk}","sum") for hk in ["min","mean","max"]
                    ] + [
                    (f"{dc}","sum") for dc in disruption_columns
                    ]
                    )
    disruptions_df["disruption_unit"] = "service/day"
    total_disruption = disruptions_df.groupby(["sector","subsector",
                        "exposure_unit",
                        "damage_cost_unit",
                        "disruption_unit"] + hazard_columns,
                        dropna=False).agg(sum_dict).reset_index()
    
    return total_disruption

def buildings_and_points_disruptions(building_df,asset_service_df,building_service_columns,hazard_columns):
    service_disruption_columns = []
    for bs in building_service_columns:
        building_df[bs] = (
                            (1 + 0.01*building_df["growth_rate_percent"]
                            )**(building_df["epoch"] - building_df["service_year"])
                            )*building_df[bs]

        for i in ["min","mean","max"]:
            building_df[f"{bs}_disrupted_{i}"] = (building_df[f"exposure_{i}"]/building_df["asset_area"])*building_df[bs]
            service_disruption_columns.append(f"{bs}_disrupted_{i}")

    # sector_effects = add_disruptions(building_df,service_disruption_columns,hazard_columns)

    percentage_disruptions = []
    epochs = list(set(building_df["epoch"].values.tolist()))
    remove_epoch = False
    for epoch in epochs:
        if "epoch" not in asset_service_df.columns.values.tolist():
            asset_service_df["epoch"] = epoch
            remove_epoch = True
        for bs in building_service_columns:
            asset_service_df[f"total_{bs}"] = (
                                                (1 + 0.01*asset_service_df["growth_rate_percent"]
                                                 )**(asset_service_df["epoch"] - asset_service_df["service_year"])
                                            )*asset_service_df[bs]

        total_service_df = asset_service_df[
                        asset_service_df["epoch"] == epoch
                        ][["asset_area"] + [f"total_{bs}" for bs in building_service_columns]].sum(axis=0)

        percentage_disruptions.append(get_percentage_exposures_losses(
                                                        building_df[building_df["epoch"] == epoch],
                                                        total_service_df,
                                                        building_service_columns))
        if remove_epoch is True:
            asset_service_df.drop("epoch",axis=1,inplace=True)
    # percentage_disruptions = pd.concat(percentage_disruptions,axis=0,ignore_index=True)
    return pd.concat(percentage_disruptions,axis=0,ignore_index=True)
    

def get_flow_paths_indexes_of_elements(flow_dataframe,path_criteria):
    tqdm.pandas()
    edge_path_index = defaultdict(list)
    for k,v in zip(chain.from_iterable(flow_dataframe[path_criteria].ravel()), flow_dataframe.index.repeat(flow_dataframe[path_criteria].str.len()).tolist()):
        edge_path_index[k].append(v)

    del flow_dataframe
    return edge_path_index

def energy_service_path_indexes(energy_service_df):
    node_path_indexes = get_flow_paths_indexes_of_elements(energy_service_df,'node_path')
    edge_path_indexes = get_flow_paths_indexes_of_elements(energy_service_df,'edge_path')

    return node_path_indexes, edge_path_indexes

def energy_criticality(asset_id,asset_type,failure_year,energy_service_df,service_growth_df,
                        supply_column,service_column):
    energy_service_df = add_service_year_and_growth_rates(
                                            energy_service_df,
                                            "destination_id",
                                            service_growth_df
                                            )

    energy_service_df["assigned_service"] = (
                                        (1 + 0.01*energy_service_df["growth_rate_percent"]
                                        )**(failure_year - energy_service_df["service_year"])
                                    )*energy_service_df[service_column]
    total_service_df = energy_service_df.drop_duplicates(subset=["destination_id"], keep="first")
    total_supply_df = energy_service_df.drop_duplicates(subset=["origin_id"], keep="first")
    total_service = total_service_df["assigned_service"].sum()
    if asset_id in total_service_df["destination_id"].values.tolist():
        critical_service = total_service_df[total_service_df["destination_id"] == asset_id]["assigned_service"].sum()
    elif asset_id in energy_service_df["origin_id"].values.tolist():
        df = total_supply_df[total_supply_df["origin_id"] == asset_id]
        critical_service = sum((df[supply_column]/df["total_installed_capacity"]))*total_service
    else:
        node_path_indexes, edge_path_indexes = energy_service_path_indexes(energy_service_df)
        if asset_type in ("nodes","areas"):
            service_df = energy_service_df[energy_service_df.index.isin(node_path_indexes[asset_id])]
        else:
            service_df = energy_service_df[energy_service_df.index.isin(edge_path_indexes[asset_id])]

        critical_service = sum((service_df[supply_column]/service_df["total_installed_capacity"])*service_df["assigned_service"])
    
    return critical_service, 100*critical_service/total_service

def get_service_columns(asset_dataframe,asset_service_columns):
    service_columns = []
    cargo_cols = [c for c in asset_service_columns if "cargo" in c]
    if cargo_cols:
        asset_dataframe["assigned_cargo"] = (1.0/365.0)*asset_dataframe[cargo_cols].astype(float).sum(axis=1)
        service_columns += ["assigned_cargo"]
    passenger_cols = [c for c in asset_service_columns if "passenger" in c]
    if passenger_cols:
        asset_dataframe["assigned_passengers"] = (1.0/365.0)*asset_dataframe[passenger_cols].astype(float).sum(axis=1)
        service_columns += ["assigned_passengers"]
    remaining_service_columns = [c for c in asset_service_columns if c not in passenger_cols + cargo_cols]
    if remaining_service_columns:
        service_columns += remaining_service_columns

    return service_columns

def add_service_year_and_growth_rates(asset_dataframe,asset_id_column,service_dataframe):
    asset_dataframe["service_year"] = service_dataframe["service_year"].values[0]
    if service_dataframe["asset_applied"].values[0] == "all":
        asset_dataframe["growth_rate_percent"] = service_dataframe["growth_rate_percent"].values[0]
    else:
        asset_dataframe["growth_rate_percent"] = 0
        for service_values in service_dataframe.itertuples():
            applied_assets = str(service_values.asset_applied).split(",")
            asset_dataframe.loc[
                        asset_dataframe[asset_id_column].isin(applied_assets),
                        "growth_rate_percent"
                        ] = service_values.growth_rate_percent

    return asset_dataframe

def modify_energy_capacities(processed_data_path,country,time_epochs,asset_service_df,scenario):
    all_capacities = pd.read_excel(os.path.join(
                                processed_data_path,
                                "data_layers",
                                "service_targets_sdgs.xlsx"),
                                sheet_name="sdg_energy_supply")
    asset_service_df["total_capacity_mw"] = asset_service_df["capacity_mw"].groupby(asset_service_df["asset_type"]).transform('sum')
    capacity_gdfs = []
    for time_epoch in time_epochs:
        installed_capacities = all_capacities[
                            (all_capacities["epoch"] == time_epoch
                            ) & (all_capacities["iso_code"] == country)
                            ]
        installed_capacities["asset_type"] = installed_capacities["asset_type"].str.lower()
        # total_installed_capacity = installed_capacities[f"{scenario}_capacity_mw"].sum()
        gdf = pd.merge(asset_service_df,
                    installed_capacities[["asset_type",f"{scenario}_capacity_mw"]],
                    how="left",on=["asset_type"]).fillna(0)
        gdf["modified_capacity"] = np.where(gdf["total_capacity_mw"] > 0, 
                                    gdf["capacity_mw"]*gdf[f"{scenario}_capacity_mw"]/gdf["total_capacity_mw"],
                                    0)
        gdf["modified_capacity"] = np.where(
                                    gdf["asset_type"] == 'solar',
                                    gdf["capacity_mw"],gdf["modified_capacity"])
        gdf.drop(["total_capacity_mw",f"{scenario}_capacity_mw"],axis=1,inplace=True)
        gdf["total_installed_capacity"] = installed_capacities[f"{scenario}_capacity_mw"].sum()
        gdf["epoch"] = time_epoch
        capacity_gdfs.append(gdf)
    return pd.concat(capacity_gdfs,axis=0,ignore_index=True)

def main(config,country,hazard_names,direct_damages_folder,
        damages_service_folder,
        network_csv,
        service_csv,
        development_scenario):

    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    baseline_year = 2023
    
    direct_damages_results = os.path.join(output_data_path,
                        f"{direct_damages_folder}_{development_scenario}")
    damages_service_results = os.path.join(output_data_path,f"{damages_service_folder}_{development_scenario}")
    if os.path.exists(damages_service_results) == False:
        os.mkdir(damages_service_results)

    asset_data_details = pd.read_csv(network_csv)
    service_data_details = pd.read_csv(service_csv)
    service_data_details = service_data_details[service_data_details["iso_code"] == country]
    service_scenarios = pd.read_excel(os.path.join(
                                processed_data_path,
                                "data_layers",
                                "service_targets_sdgs.xlsx"),
                                sheet_name="sdg_applied_goals")
    service_scenarios = service_scenarios[service_scenarios["iso_code"] == country.upper()]
    
    energy_demand_factor = 1.0
    energy_flow_assessment = True
    for asset_info in asset_data_details.itertuples():
        asset_id = asset_info.asset_id_column
        asset_service_columns = str(asset_info.service_columns).split(",")
        demand_effect_df = service_scenarios[
                            (service_scenarios["constraint_type"] == "demand"
                                ) & (
                            service_scenarios["asset_layer"] == asset_info.asset_gpkg)] 
        demand_change_factor = demand_effect_df.iloc[0][
                            f"future_scenario_{development_scenario}"
                            ]/service_scenarios.iloc[0][f"future_scenario_bau"]
        
        damage_file = os.path.join(
                                    direct_damages_results,
                                    f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages.csv"
                                    )
        if os.path.isfile(damage_file) is True:
            damage_results = pd.read_csv(damage_file)
            if os.path.isfile(os.path.join(processed_data_path,
                                "infrastructure",
                                asset_info.sector,
                                f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}.csv")) is True:
                asset_service_df = pd.read_csv(os.path.join(
                                            processed_data_path,
                                            "infrastructure",
                                            asset_info.sector,
                                            f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}.csv")
                                        )
                asset_service_df["asset_area"] = 1
            else:
                asset_service_df = gpd.read_file(os.path.join(
                                            processed_data_path,
                                            "infrastructure",
                                            asset_info.sector,
                                            f"{country}_{asset_info.asset_gpkg}.gpkg"),
                                        layer=asset_info.asset_layer)
                if asset_info.asset_layer == "areas":
                    asset_service_df = asset_service_df.to_crs(epsg=epsg_caribbean)
                    asset_service_df["asset_area"] = asset_service_df.geometry.area
                elif asset_info.asset_layer == "edges":
                    asset_service_df = asset_service_df.to_crs(epsg=epsg_caribbean)
                    asset_service_df["asset_area"] = asset_service_df.geometry.length
                else:
                    asset_service_df["asset_area"] = 1

            if asset_info.service_disruption_level == "asset":
                hazard_columns = [h for h in damage_results.columns.values.tolist() if h in hazard_names]
                service_columns = get_service_columns(asset_service_df,asset_service_columns)
                if asset_info.asset_gpkg == "energy":
                    asset_service_df = asset_service_df[asset_service_df["development_scenario"] == development_scenario]
                    merge_columns = [asset_id,"epoch"]
                else:
                    merge_columns = [asset_id]
                asset_service_df = asset_service_df[merge_columns + ["asset_area"] + service_columns]
                asset_service_df[service_columns] = asset_service_df[service_columns].fillna(0)
                asset_service_df[service_columns] = demand_change_factor*asset_service_df[service_columns]
                asset_service_df = add_service_year_and_growth_rates(
                                            asset_service_df,
                                            asset_id,
                                            service_data_details[
                                                service_data_details["asset_gpkg"] == asset_info.asset_gpkg
                                                ]
                                            )
                damage_results = pd.merge(damage_results,asset_service_df,how="left",on=merge_columns)
                sector_damages_losses = buildings_and_points_disruptions(damage_results,
                                                        asset_service_df,
                                                        service_columns,
                                                        hazard_columns)
                sector_damages_losses.to_csv(os.path.join(damages_service_results,
                            f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages_losses.csv"),index=False)

            elif asset_info.asset_gpkg == "energy":
                if energy_flow_assessment is True:
                    energy_demand_factor = demand_change_factor                
                    energy_flow_df = pd.read_parquet(os.path.join(
                                    output_data_path,
                                    "energy_flow_paths",
                                    f"{country}_energy_paths.parquet"))
                    # node_path_indexes, edge_path_indexes = energy_service_path_indexes(energy_flow_df)
                    energy_flow_df["pop_2020"] = energy_demand_factor*energy_flow_df["pop_2020"]
                    modified_flow_df = energy_flow_df.drop_duplicates(
                                        subset=["origin_id"], keep="first")[["origin_id","asset_type","capacity_mw"]]
                    flow_dfs = []
                    for time in list(set(damage_results["epoch"].values.tolist())):
                        modified_flow_df = modify_energy_capacities(processed_data_path,
                                                                    country,
                                                                    [time],
                                                                    modified_flow_df,development_scenario)
                        flow_df = pd.merge(energy_flow_df,
                                    modified_flow_df[["origin_id","modified_capacity","total_installed_capacity","epoch"]],
                                    how="left",on=["origin_id"]).fillna(0)
                        total_service = flow_df.drop_duplicates(subset=["destination_id"], keep="first")["pop_2020"].sum()
                        flow_df["customers_disrupted_percentage"] = 100*(flow_df["pop_2020"]*flow_df["modified_capacity"]/flow_df["total_installed_capacity"])/total_service
                        flow_df = flow_df.sort_values(by=["customers_disrupted_percentage"],ascending=False)
                        flow_df["customers_disrupted_percentage_cumsum"] = flow_df["customers_disrupted_percentage"].cumsum()
                        flow_dfs.append(flow_df)
        
                    flow_dfs = pd.concat(flow_dfs,axis=0,ignore_index=True)
                    del energy_flow_df
            
                    flow_dfs.to_csv(os.path.join(damages_service_results,
                                f"{country}_{asset_info.asset_gpkg}_asset_damages_losses.csv"),index=False)
                    flow_dfs.to_parquet(os.path.join(damages_service_results,
                                f"{country}_{asset_info.asset_gpkg}_asset_damages_losses.parquet"),index=False)
                    energy_flow_assessment = False
            
            elif asset_info.asset_gpkg == "roads":
                path_types = [
                                "health_pathdata_time_m_60.parquet",
                                "schools_pathdata_time_m_60.parquet",
                                "pathdata_time_m_30.parquet"]
                truncate = [False,False,False]
                path_dfs = []
                total_trips = 0
                for idx,(pt,tr) in enumerate(zip(path_types,truncate)): 
                    path_df = pd.read_parquet(os.path.join(output_data_path,
                                        'transport',
                                        'path and flux data',
                                        f'{country.upper()}_{pt}'))
                    if tr is True:
                        path_df, *_ = tfdf.truncate_by_threshold(path_df, threshold=0.99)
                    path_df["flux"] = demand_change_factor*path_df["flux"]
                    total_trips += path_df['flux'].sum()
                    path_dfs.append(path_df)
                
                path_dfs = pd.concat(path_dfs,axis=0,ignore_index=True)
                path_dfs["trips_disrupted_percentage"] = 100.0*path_dfs["flux"]/total_trips
                path_dfs = path_dfs.sort_values(by=["trips_disrupted_percentage"],ascending=False)
                path_dfs["trips_disrupted_percentage_cumsum"] = path_dfs["trips_disrupted_percentage"].cumsum()
                        

                path_dfs.to_csv(os.path.join(damages_service_results,
                            f"{country}_roads_asset_damages_losses.csv"),index=False)
                path_dfs.to_parquet(os.path.join(damages_service_results,
                            f"{country}_roads_asset_damages_losses.parquet"),index=False)

if __name__ == "__main__":
    CONFIG = load_config()
    try:
        country =  str(sys.argv[1])
        hazard_names = ast.literal_eval(str(sys.argv[2]))
        direct_damages_folder = str(sys.argv[3])
        damages_service_folder = str(sys.argv[4])
        network_csv = str(sys.argv[5])
        service_csv = str(sys.argv[6])
        development_scenario = str(sys.argv[7])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()

    main(CONFIG,country,hazard_names,direct_damages_folder,
        damages_service_folder,
        network_csv,
        service_csv,
        development_scenario)
