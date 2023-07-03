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
    for hk in ["amin","mean","amax"]:
        damages_df[f"exposure_{hk}_percentage"] = 100.0*damages_df[f"exposure_{hk}"]/sector_total_service["asset_area"]
    for sc in service_disruption_columns:
        for hk in ["amin","mean","amax"]: 
            damages_df[f"{sc}_disrupted_{hk}_percentage"] = 100.0*damages_df[f"{sc}_disrupted_{hk}"]/sector_total_service[f"total_{sc}"]

    return damages_df


def add_disruptions(disruptions_df,disruption_columns,hazard_columns):
    sum_dict = dict([
                    (f"exposure_{hk}","sum") for hk in ["amin","mean","amax"]
                    ]+[
                    (f"damage_{hk}","sum") for hk in ["amin","mean","amax"]
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

        for i in ["amin","mean","amax"]:
            building_df[f"{bs}_disrupted_{i}"] = (building_df[f"exposure_{i}"]/building_df["asset_area"])*building_df[bs]
            if "asset_fix" in building_df.columns.values.tolist():
                building_df[f"{bs}_disrupted_{i}"] = np.where(building_df["asset_fix"] == 1, 0, building_df[f"{bs}_disrupted_{i}"])
            service_disruption_columns.append(f"{bs}_disrupted_{i}")
    sector_effects = add_disruptions(building_df,service_disruption_columns,hazard_columns)

    percentage_disruptions = []
    epochs = list(set(building_df["epoch"].values.tolist()))
    for epoch in epochs:
        asset_service_df["epoch"] = epoch
        for bs in building_service_columns:
            asset_service_df[f"total_{bs}"] = (
                                                (1 + 0.01*asset_service_df["growth_rate_percent"]
                                                 )**(asset_service_df["epoch"] - asset_service_df["service_year"])
                                            )*asset_service_df[bs]

        total_service_df = asset_service_df[["asset_area"] + [f"total_{bs}" for bs in building_service_columns]].sum(axis=0)

        percentage_disruptions.append(get_percentage_exposures_losses(
                                                        sector_effects[sector_effects["epoch"] == epoch],
                                                        total_service_df,
                                                        building_service_columns))
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

def energy_disruptions(asset_failure_set,supply_loss_set,failure_year,energy_edges,energy_service_df,
                        service_growth_df,
                        supply_column,service_column):
    
    node_path_indexes, edge_path_indexes = energy_service_path_indexes(energy_service_df)

    if str(asset_failure_set) != "nan":
        failure_indexes = list(set(list(chain.from_iterable([path_idx for path_key,path_idx in node_path_indexes.items() if path_key in asset_failure_set]))))
        failure_indexes = list(set(failure_indexes + list(chain.from_iterable([path_idx for path_key,path_idx in edge_path_indexes.items() if path_key in asset_failure_set]))))
        if len(failure_indexes) > 0:
            energy_service_df = add_service_year_and_growth_rates(
                                            energy_service_df,
                                            "destination_id",
                                            service_growth_df
                                            )

            energy_service_df["assigned_service"] = (
                                                (1 + 0.01*energy_service_df["growth_rate_percent"]
                                                )**(failure_year - energy_service_df["service_year"])
                                            )*energy_service_df[service_column]
            total_service = energy_service_df.drop_duplicates(subset=["destination_id"], keep="first")["assigned_service"].sum()
            # total_supply = energy_service_df.drop_duplicates(subset=["origin_id"], keep="first")[supply_column].sum()
            total_supply = energy_service_df["total_installed_capacity"].values[0] 
            failed_supply = sum(supply_loss_set)
            failed_demand_nodes = list(set([n for n in asset_failure_set if n in list(set(energy_service_df["destination_id"].values.tolist()))]))
            failed_service_df = energy_service_df[energy_service_df.index.isin(failure_indexes)]
            if failed_supply == total_supply:
                return total_service, 100.0
            elif len(failed_demand_nodes) == len(list(set(energy_service_df["destination_id"].values.tolist()))):
                return total_service, 100.0
            else:
                if sum(supply_loss_set) > 0:
                    failed_supply_df = pd.DataFrame(list(zip(asset_failure_set,supply_loss_set)),columns=["origin_id","capacity_loss"])
                    failed_supply_df = failed_supply_df[failed_supply_df["capacity_loss"]>0]
                    energy_service_df = pd.merge(energy_service_df,failed_supply_df,how="left",on=["origin_id"]).fillna(0)
                    energy_service_df["supply"] = energy_service_df[supply_column] - energy_service_df["capacity_loss"] 
                    # print (energy_service_df)
                else:
                    energy_service_df["supply"] = energy_service_df[supply_column].copy()
                
                supply_df = energy_service_df[["origin_id",supply_column,"supply"]].drop_duplicates(subset=["origin_id"], keep="first")
                demand_df = energy_service_df[["destination_id","assigned_service"]].drop_duplicates(subset=["destination_id"], keep="first")
                od_matrix = supply_df.merge(demand_df, how='cross')
                od_matrix = od_matrix[~(od_matrix.origin_id.isin(asset_failure_set) | od_matrix.destination_id.isin(asset_failure_set))]
                # del supply_df, demand_df
                edges = energy_edges[~energy_edges["edge_id"].isin(asset_failure_set)]
                edges = edges[~(edges["from_node"].isin(asset_failure_set) | edges["to_node"].isin(asset_failure_set))]
                # print (edges)
                remaining_service_df = network_ods_assembly(od_matrix,edges,
                                None,["supply","assigned_service"],directed=False)
                # remaining_service_df = energy_service_df[~energy_service_df.index.isin(failure_indexes)]
                if len(remaining_service_df.index) > 0:
                    # print (remaining_service_df)
                    remaining_service_df = remaining_service_df.groupby(["destination_id","assigned_service"])["supply"].sum().reset_index()
                    disrupted_service = total_service - sum((1.0*remaining_service_df["supply"]/total_supply)*remaining_service_df["assigned_service"])
                else:
                    disrupted_service = total_service*(supply_df[supply_column].sum())/total_supply
                return disrupted_service, 100.0*disrupted_service/total_service
        else:
            return 0, 0.0
    else:
        return 0, 0.0

# def modify_epoch(damage_dataframe,baseline_year):
#     damage_dataframe["epoch"] = damage_dataframe["epoch"].fillna(baseline_year)
#     damage_dataframe.loc[damage_dataframe["epoch"] < baseline_year,"epoch"] = baseline_year
#     return damage_dataframe

def get_service_columns(asset_dataframe,asset_service_columns):
    service_columns = []
    cargo_cols = [c for c in asset_service_columns if "cargo" in c]
    if cargo_cols:
        asset_dataframe["assigned_cargo"] = 1.0/365*asset_dataframe[cargo_cols].sum(axis=1)
        service_columns += ["assigned_cargo"]
    passenger_cols = [c for c in asset_service_columns if "passenger" in c]
    if passenger_cols:
        asset_dataframe["assigned_passengers"] = 1.0/365*asset_dataframe[passenger_cols].sum(axis=1)
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
        development_scenario,
        adaptation_scenario):

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
    service_scenarios = pd.read_excel(os.path.join(
                                processed_data_path,
                                "data_layers",
                                "service_targets_sdgs.xlsx"),
                                sheet_name="sdg_applied_goals")
    service_scenarios = service_scenarios[service_scenarios["iso_code"] == country.upper()]

    energy_sector_results = []
    energy_demand_factor = 1.0
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
        
        if adaptation_scenario == "no_adaptation":
            damage_file = os.path.join(
                                        direct_damages_results,
                                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages.csv"
                                        )
        else:
            damage_file = os.path.join(
                                        direct_damages_results,
                                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_targets_costs.csv"
                                        )


        if os.path.isfile(damage_file) is True:
            # damage_results = pd.read_parquet(damage_file)
            damage_results = pd.read_csv(damage_file)
            # damage_results = modify_epoch(damage_results,baseline_year)
            # replicate and add landslides 
            # landslide = damage_results[damage_results["hazard"] == "landslide"]
            # landslides = [landslide]
            # for epoch in [2030,2050]:
            #     l = landslide.copy()
            #     l["epoch"] = epoch
            #     landslides.append(l)
            # landslides = pd.concat(landslides,axis=0,ignore_index=True)
            # damage_results = damage_results[damage_results["hazard"] != "landslide"]
            # damage_results = pd.concat([damage_results,landslides],axis=0,ignore_index=True)
            # del landslide, l, landslides
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

            if asset_info.asset_gpkg == "energy":
                energy_demand_factor = demand_change_factor
                if "capacity_mw" in asset_service_df.columns.values.tolist():
                    asset_service_dfs = modify_energy_capacities(
                                        processed_data_path,
                                        country,
                                        list(set(damage_results["epoch"].values.tolist())),
                                        asset_service_df,development_scenario)
                    damage_results = pd.merge(damage_results,
                        asset_service_dfs[[asset_id,"epoch","modified_capacity","total_installed_capacity","asset_area"]],
                        how="left",on=[asset_id,"epoch"])
                else:
                    damage_results = pd.merge(damage_results,asset_service_df[[asset_id,"asset_area"]],how="left",on=[asset_id])
                    damage_results["modified_capacity"] = 0
                    damage_results["total_installed_capacity"] = 0
                rename_dict = dict([(asset_id,"asset_id")]+[(f"exposure_{hk}",f"{asset_info.asset_layer}_exposure_{hk}") for hk in ["amin","mean","amax"]])
                damage_results.rename(columns=rename_dict,inplace=True)
                total_area = asset_service_df[asset_service_df["asset_type"] != "dummy"]["asset_area"].sum()
                for hk in ["amin","mean","amax"]:
                    damage_results[
                            f"{asset_info.asset_layer}_exposure_{hk}_percentage"
                            ] = 100.0*damage_results[f"{asset_info.asset_layer}_exposure_{hk}"]/total_area
                    damage_results[
                                f"generation_capacity_loss_{hk}"
                                ] = damage_results[
                                    "modified_capacity"
                                    ]*damage_results[
                                        f"{asset_info.asset_layer}_exposure_{hk}"
                                        ]/damage_results["asset_area"]
                energy_sector_results.append(damage_results)
            elif asset_info.asset_gpkg == "roads":
                # load roads, roads network
                roads, road_net = tfdf.get_roads(os.path.join(processed_data_path,
                                            'infrastructure',
                                            'transport'), country, ['edge_id', 'length_m', 'time_m'])
                roads = roads.to_crs(epsg=epsg_caribbean)
                total_length = roads['length_m'].sum()
                """Will skip reading the whole parquet, as it is too big!!!
                # step 1: get path and flux dataframe and save
                path_df = pd.read_parquet(os.path.join(output_data_path,
                                    'transport',
                                    'path and flux data',
                                    f'{country.upper()}_pathdata_time_m_30.parquet'),
                                    engine="fastparquet")
                # step 1(b): truncate disruption to remove smallest fluxes
                path_df, *_ = tfdf.truncate_by_threshold(path_df, threshold=TRUNC_THRESH)

                path_df.to_parquet(os.path.join(output_data_path,
                                    'transport',
                                    'path and flux data',
                                    f'{country.upper()}_pathdata_time_m_30_truncated.parquet'),
                                    index=False)
                """
                hazard_columns = [h for h in damage_results.columns.values.tolist() if h in hazard_names]
                for hk in ["amin","mean","amax"]:
                    damage_results[
                            f"exposure_{hk}_percentage"
                            ] = 100.0*damage_results[f"exposure_{hk}"]/total_length

                exposure_damage_dict = dict(
                                [
                                    (f"exposure_{hk}","sum") for hk in ["amin","mean","amax"]
                                ]+ [
                                    (f"exposure_{hk}_percentage","sum") for hk in ["amin","mean","amax"]
                                ]+[
                                    (f"damage_{hk}","sum") for hk in ["amin","mean","amax"]
                                ])
                disruption_df = damage_results.groupby(["sector","subsector",
                                    "exposure_unit",
                                    "damage_cost_unit"] + hazard_columns,
                                    dropna=False).agg(exposure_damage_dict).reset_index()
                for hk in ["amin","mean","amax"]:
                    df = damage_results[damage_results[f"damage_{hk}"]>0]
                    if "asset_fix" in damage_results.columns.values.tolist():
                        df = df[df["asset_fix"] == 0]
                    # print (df)
                    df = df.groupby(["sector","subsector",
                                    "exposure_unit",
                                    "damage_cost_unit"] + hazard_columns,dropna=False)[asset_id].apply(list).reset_index(name=f"asset_set_{hk}")
                    disruption_df = pd.merge(disruption_df,df,how="left", on=["sector","subsector",
                                                                            "exposure_unit",
                                                                            "damage_cost_unit"] + hazard_columns)
                    del df
                # disruption_df.to_csv("test.csv")
                del damage_results
                path_types = [
                                "health_pathdata_time_m_60.parquet",
                                "schools_pathdata_time_m_60.parquet",
                                "pathdata_time_m_30_truncated.parquet"]
                truncate = [False,True,False]
                path_dfs = []
                total_trips = 0
                for idx,(pt,tr) in enumerate(zip(path_types,truncate)): 
                    path_df = pd.read_parquet(os.path.join(output_data_path,
                                        'transport',
                                        'path and flux data',
                                        f'{country.upper()}_{pt}'),
                                        engine="fastparquet")
                    if tr is True:
                        path_df, *_ = tfdf.truncate_by_threshold(path_df, threshold=0.99)
                    path_df["flux"] = demand_change_factor*path_df["flux"]
                    total_trips += path_df['flux'].sum()
                    path_dfs.append(path_df)
                
                path_dfs = pd.concat(path_dfs,axis=0,ignore_index=True)

                # step 2: model disruption
                disrupt_dfs = tfdf.get_disruption_stats(disruption_df.copy(), path_dfs,road_net, COST)
                disrupt_dfs.drop([f"asset_set_{hk}" for hk in ["amin","mean","amax"]],axis=1,inplace=True)
                # del loss_df, path_df

                trip_loss_dict = dict(
                                [
                                    (f"trips_disrupted_{hk}","sum") for hk in ["amin","mean","amax"]
                                ]+ [
                                    (f"trips_rerouted_{hk}","sum") for hk in ["amin","mean","amax"]
                                ]+ [
                                    (f"time_m_delta_{hk}","sum") for hk in ["amin","mean","amax"]
                                ])
                disrupt_dfs = disrupt_dfs.groupby(["sector","subsector",
                                        "exposure_unit",
                                        "damage_cost_unit"] + hazard_columns,
                                        dropna=False).agg(trip_loss_dict).reset_index()
                for hk in ["amin","mean","amax"]:
                    disrupt_dfs[
                            f"trips_disrupted_{hk}_percentage"
                            ] = 100.0*disrupt_dfs[f"trips_disrupted_{hk}"]/total_trips
                    disrupt_dfs[
                            f"trips_rerouted_{hk}_percentage"
                            ] = 100.0*disrupt_dfs[f"trips_rerouted_{hk}"]/total_trips
                disruption_df.drop([f"asset_set_{hk}" for hk in ["amin","mean","amax"]],axis=1,inplace=True)
                disruption_df = pd.merge(disruption_df,disrupt_dfs,how="left", on=["sector","subsector",
                                                                            "exposure_unit",
                                                                            "damage_cost_unit"] + hazard_columns)
                del disrupt_dfs
                disruption_df = add_service_year_and_growth_rates(disruption_df,
                                                        None,service_data_details[
                                                        service_data_details["asset_gpkg"] == asset_info.asset_gpkg
                                                        ])
                for hk in ["amin","mean","amax"]:
                    disruption_df[f"trips_disrupted_{hk}"] = (
                                                    (1 + 0.01*disruption_df["growth_rate_percent"]
                                                    )**(disruption_df["epoch"] - disruption_df["service_year"])
                                                )*disruption_df[f"trips_disrupted_{hk}"]
                    disruption_df[f"trips_rerouted_{hk}"] = (
                                                    (1 + 0.01*disruption_df["growth_rate_percent"]
                                                    )**(disruption_df["epoch"] - disruption_df["service_year"])
                                                )*disruption_df[f"trips_rerouted_{hk}"]
                disruption_df.drop(["growth_rate_percent","service_year"],axis=1,inplace=True)
                disruption_df.to_csv(os.path.join(damages_service_results,
                            f"{country}_roads_edges_sector_damages_losses.csv"),index=False)

            elif asset_info.service_disruption_level == "asset":
                hazard_columns = [h for h in damage_results.columns.values.tolist() if h in hazard_names]
                service_columns = get_service_columns(asset_service_df,asset_service_columns)
                asset_service_df = asset_service_df[[asset_id, "asset_area"] + service_columns]
                asset_service_df[service_columns] = asset_service_df[service_columns].fillna(0)
                asset_service_df[service_columns] = demand_change_factor*asset_service_df[service_columns]
                asset_service_df = add_service_year_and_growth_rates(
                                            asset_service_df,
                                            asset_id,
                                            service_data_details[
                                                service_data_details["asset_gpkg"] == asset_info.asset_gpkg
                                                ]
                                            )
                damage_results = pd.merge(damage_results,asset_service_df,how="left",on=[asset_id])
                sector_damages_losses = buildings_and_points_disruptions(damage_results,
                                                        asset_service_df,
                                                        service_columns,
                                                        hazard_columns)
                sector_damages_losses.to_csv(os.path.join(damages_service_results,
                            f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_sector_damages_losses.csv"),index=False)

    if energy_sector_results:
        energy_edges = gpd.read_file(os.path.join(
                                        processed_data_path,
                                        "infrastructure",
                                        "energy",
                                        f"{country}_energy.gpkg"),
                                    layer="edges")[['from_node','to_node','edge_id']]

        energy_sector_results = pd.concat(energy_sector_results,axis=0,ignore_index=True)
        energy_flow_df = pd.read_parquet(os.path.join(
                                output_data_path,
                                "energy_flow_paths",
                                f"{country}_energy_paths.parquet"))
        energy_flow_df["pop_2020"] = energy_demand_factor*energy_flow_df["pop_2020"]
        modified_flow_df = energy_flow_df.drop_duplicates(
                            subset=["origin_id"], keep="first")[["origin_id","asset_type","capacity_mw"]]
        flow_dfs = []
        for time in list(set(energy_sector_results["epoch"].values.tolist())):
            modified_flow_df = modify_energy_capacities(processed_data_path,
                                                        country,
                                                        [time],
                                                        modified_flow_df,development_scenario)
            flow_dfs.append(pd.merge(energy_flow_df,
                        modified_flow_df[["origin_id","modified_capacity","total_installed_capacity","epoch"]],
                        how="left",on=["origin_id"]))
        
        flow_dfs = pd.concat(flow_dfs,axis=0,ignore_index=True)
        del energy_flow_df
        energy_sector_results["disruption_unit"] = "customers/day"
        hazard_columns = [h for h in energy_sector_results.columns.values.tolist() if h in hazard_names]
        # sum_dict = dict(
        #                 [(hk,"sum") for hk in energy_sector_results.columns.values.tolist() if "_exposure_" in hk
        #                 ]+[(f"damage_{hk}","sum") for hk in ["amin","mean","amax"]
        #                 ]+[(f"generation_capacity_loss_{hk}","sum") for hk in ["amin","mean","amax"]]
        #                 )
        sum_dict = dict(
                        [(hk,"sum") for hk in energy_sector_results.columns.values.tolist() if "_exposure_" in hk
                        ]+[(f"damage_{hk}","sum") for hk in ["amin","mean","amax"]
                        ])
        total_disruption = energy_sector_results.groupby(["sector","subsector",
                        "damage_cost_unit",
                        "disruption_unit"] + hazard_columns,
                        dropna=False).agg(sum_dict).reset_index()
        for hk in ["amin","mean","amax"]:
            df = energy_sector_results[energy_sector_results[f"damage_{hk}"]>0]
            if "asset_fix" in df.columns.values.tolist():
                df = df[df["asset_fix"] == 0]
            
            df0 = df.groupby(["sector","subsector",
                        "damage_cost_unit",
                        "disruption_unit"] + hazard_columns,
                        dropna=False).agg(
                        dict([(f"generation_capacity_loss_{hk}","sum")])).reset_index()
            # breakpoint()
            df1 = df.groupby(["sector","subsector",
                            "damage_cost_unit",
                            "disruption_unit"] + hazard_columns,dropna=False)["asset_id"].apply(list).reset_index(name=f"asset_set_{hk}")
            df2 = df.groupby(["sector","subsector",
                            "damage_cost_unit",
                            "disruption_unit"] + hazard_columns,dropna=False)[f"generation_capacity_loss_{hk}"].apply(list).reset_index(name=f"generation_set_{hk}")
            # df1['precipitation_factor'] = df1['precipitation_factor'].astype(total_disruption['precipitation_factor'].dtype)
            # df2['precipitation_factor'] = df2['precipitation_factor'].astype(total_disruption['precipitation_factor'].dtype)
            for c in df0.columns.values.tolist():
                if c in total_disruption.columns.values.tolist():
                    df0[c] = df0[c].astype(total_disruption[c].dtype)

            for c in df1.columns.values.tolist():
                if c in total_disruption.columns.values.tolist():
                    df1[c] = df1[c].astype(total_disruption[c].dtype)

            for c in df2.columns.values.tolist():
                if c in total_disruption.columns.values.tolist():
                    df2[c] = df2[c].astype(total_disruption[c].dtype)

            total_disruption = pd.merge(total_disruption,df0,how="left", on=["sector","subsector",
                                                                    "damage_cost_unit","disruption_unit"] + hazard_columns)
            total_disruption = pd.merge(total_disruption,df1,how="left", on=["sector","subsector",
                                                                    "damage_cost_unit","disruption_unit"] + hazard_columns)
            total_disruption = pd.merge(total_disruption,df2,how="left", on=["sector","subsector",
                                                                    "damage_cost_unit","disruption_unit"] + hazard_columns)
            del df, df0, df1, df2
            total_disruption["customer_losses"] = total_disruption.progress_apply(
                                                lambda x:energy_disruptions(
                                                    x[f"asset_set_{hk}"],x[f"generation_set_{hk}"],x["epoch"],
                                                    energy_edges,
                                                    flow_dfs[flow_dfs["epoch"] == x.epoch],
                                                    service_data_details[
                                                        service_data_details["asset_gpkg"] == asset_info.asset_gpkg
                                                        ],
                                                    "modified_capacity","pop_2020"),
                                                axis=1)
            total_disruption[[f"customers_disrupted_{hk}",f"customers_disrupted_{hk}_percentage"]] = total_disruption["customer_losses"].apply(pd.Series)
            total_disruption.drop(["customer_losses"],axis=1,inplace=True)
            total_disruption.drop([f"asset_set_{hk}",f"generation_set_{hk}"],axis=1,inplace=True)
        
        total_disruption.to_csv(os.path.join(damages_service_results,
                            f"{country}_{energy_sector_results.subsector.values[0]}_sector_damages_losses.csv"),index=False)




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
        adaptation_scenario = str(sys.argv[8])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()

    main(CONFIG,country,hazard_names,direct_damages_folder,
        damages_service_folder,
        network_csv,
        service_csv,
        development_scenario,adaptation_scenario)
