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
def main(config,country,
        grouping_columns,
        direct_damages_folder,
        network_csv,
        development_scenario):

    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    baseline_year = 2023
    # edge_id sector  subsector   asset_layer exposure_unit   damage_cost_unit    hazard  isoa3   epoch   rcp rp
    direct_damages_results = os.path.join(output_data_path,
                        f"{direct_damages_folder}_{development_scenario}")
    asset_data_details = pd.read_csv(network_csv)
    damage_sum_dict = dict(
                    [
                        (f"damage_{hk}","sum") for hk in ["min","mean","max"]
                    ]
                )

    cost_sum_dict = dict(
                    [
                        (f"adaptation_investment_{hk}","sum") for hk in ["min","mean","max"]
                    ]
                )
    sector_damages = []
    sector_adaptation_costs = []
    for asset_info in asset_data_details.itertuples():
        asset_id = asset_info.asset_id_column
        damage_file = os.path.join(
                                    direct_damages_results,
                                    f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages.csv"
                                    )
        if os.path.isfile(damage_file) is True:
            damage_results = pd.read_csv(damage_file)
            damage_results = damage_results[damage_results.hazard != "fluvial_undefended"]
            damage_results = damage_results.groupby(
                                grouping_columns,
                                dropna=False
                                ).agg(damage_sum_dict).reset_index()
            sector_damages.append(damage_results)

        if country == "xyz":
            cost_file = os.path.join(
                                        direct_damages_results,
                                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_targets_costs.csv"
                                    )
            parquet = False
        else:
            cost_file = os.path.join(
                                        direct_damages_results,
                                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_targets_costs.parquet"
                                    )
            parquet = True

        if os.path.isfile(cost_file) is True:
            if parquet == False:
                cost_results = pd.read_csv(cost_file) 
            else:
                cost_results = pd.read_parquet(cost_file)
            cost_results = cost_results[
                            (
                                cost_results["service_resilience_target_percentage"] == 100
                                ) & (
                                cost_results["asset_fix"] == 1)
                            ]
            cost_results = cost_results[cost_results.hazard != "fluvial_undefended"]
            flood_costs = cost_results[cost_results.hazard.isin(["coastal","fluvial_defended","pluvial"])]
            if len(flood_costs.index) > 0:
                flood_costs = flood_costs.sort_values(
                                    by=["adaptation_investment_min",
                                        "adaptation_investment_mean",
                                        "adaptation_investment_max"],ascending=False)
                flood_costs = flood_costs.drop_duplicates(subset=[asset_id] + grouping_columns,keep="first")
                no_flood_costs = cost_results[~cost_results.hazard.isin(["coastal","fluvial_defended","pluvial"])]
                cost_results = pd.concat([flood_costs,no_flood_costs],axis=0,ignore_index=True)

            cost_results = cost_results.groupby(
                                grouping_columns,
                                dropna=False
                                ).agg(cost_sum_dict).reset_index()
            sector_adaptation_costs.append(cost_results)


    sector_damages = pd.concat(sector_damages,axis=0,ignore_index=True)
    sector_damages = sector_damages.groupby(
                                grouping_columns,
                                dropna=False
                                ).agg(damage_sum_dict).reset_index()
    sector_damages.to_csv(os.path.join(direct_damages_results,
                            f"{country}_combined_asset_damages.csv"),index=False)

    if len(sector_adaptation_costs) > 0:
        sector_adaptation_costs = pd.concat(sector_adaptation_costs,axis=0,ignore_index=True)
        sector_adaptation_costs = sector_adaptation_costs.groupby(
                                    grouping_columns,
                                    dropna=False
                                    ).agg(cost_sum_dict).reset_index()
        sector_adaptation_costs.to_csv(os.path.join(direct_damages_results,
                                f"{country}_combined_asset_costs.csv"),index=False)



if __name__ == "__main__":
    CONFIG = load_config()
    try:
        country =  str(sys.argv[1])
        grouping_columns = ast.literal_eval(str(sys.argv[2]))
        direct_damages_folder = str(sys.argv[3])
        network_csv = str(sys.argv[4])
        development_scenario = str(sys.argv[5])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()

    main(CONFIG,country,
        grouping_columns,
        direct_damages_folder,
        network_csv,
        development_scenario)
