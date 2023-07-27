"""Estimate direct damages to physical assets exposed to hazards

"""
import sys
import os

import pandas as pd
import geopandas as gpd
import itertools
import numpy as np
import ast
import warnings
from add_costs import add_costs, get_adaptation_uplifts_reductions
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from analysis_utils import *
from tqdm import tqdm
tqdm.pandas()

epsg_caribbean = 32620

def get_damage_exposure_values(df1,df2,merge_columns,hk,adapt_damage_file):
    if len(df2) > 0:
        df2 = pd.concat(df2,axis=0,ignore_index=True)
        if os.path.isfile(adapt_damage_file) is True:
            adapt_damage_df = pd.read_csv(adapt_damage_file)
            df2 = pd.merge(df2,adapt_damage_df,
                                how="left",
                                on=merge_columns)
            df2[f"damage_{hk}"] = df2[f"damage_{hk}"].fillna(0)
            df2[f"exposure_{hk}"] = df2[f"exposure_{hk}"].fillna(0)
            ignore_columns = [
                    f"{d}_{e}" for idx, (d,e) in enumerate(
                                list(
                                    itertools.product(
                                        ["damage","exposure"], 
                                        [h for h in ["min","mean","max"] if h != hk]
                                        )
                                    )
                                )]
            df2.drop(ignore_columns,axis=1,inplace=True)
        else:
            df2[f"damage_{hk}"] = 0
            df2[f"exposure_{hk}"] = 0
        # breakpoint()
        target_index_columns = [c for c in df2.columns.values.tolist() if c not in [f"damage_{hk}",f"exposure_{hk}"]]
        df2 = df2.drop_duplicates(subset=target_index_columns,keep="first")
        df1.append(df2.set_index(target_index_columns))
    return df1

def add_adaptation_costs(geo_dataframe,dataframe_type,
                cost_column,
                cost_uplift_column,epsg=epsg_caribbean):
    # geo_dataframe = gpd.GeoDataFrame(dataframe,
    #                             geometry = 'geometry',
    #                             crs={'init': f'epsg:{epsg}'})
    # if dataframe_type == 'edges':
    #     geo_dataframe['dimension'] = geo_dataframe.geometry.length
    # elif dataframe_type == 'areas':
    #     geo_dataframe['dimension'] = geo_dataframe.geometry.area
    # else:
    #     geo_dataframe['dimension'] = 1
    # # geo_dataframe.drop('geometry',axis=1,inplace=True)

    geo_dataframe["cost_markup"] = np.where(geo_dataframe["rebuild_asset"] == "yes",1,0) + geo_dataframe[cost_uplift_column]
    geo_dataframe[
            [f"adaptation_investment_{hk}" for hk in ["min","max"]]
            ] = geo_dataframe[
                    [f"{cost_column}_{hk}" for hk in ["min","max"]]
                    ].multiply(geo_dataframe["cost_markup"],axis="index")
    geo_dataframe["adaptation_investment_mean"] = 0.5*(geo_dataframe["adaptation_investment_min"] + geo_dataframe["adaptation_investment_max"])
    
    return geo_dataframe

def main(config,country,hazard_columns,direct_damages_folder,
        damages_service_folder,
        network_csv,
        development_scenario):

    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    baseline_year = 2023
    
    no_adaptation_results = os.path.join(output_data_path,
                                "no_adaptation",
                                f"damage_service_losses_{development_scenario}")

    direct_damages_results = os.path.join(output_data_path,
                        f"{direct_damages_folder}_{development_scenario}")
    damages_service_results = os.path.join(output_data_path,f"{damages_service_folder}_{development_scenario}")
    if os.path.exists(damages_service_results) == False:
        os.mkdir(damages_service_results)

    asset_data_details = pd.read_csv(network_csv)
    
    reduction_df = get_adaptation_uplifts_reductions()
    hazard_mapping = pd.read_csv(os.path.join(processed_data_path,
                            "damage_curves",
                            "hazard_damage_parameters.csv"))[["hazard","hazard_type"]]
    reduction_df = pd.merge(hazard_mapping,reduction_df,how="left",on=["hazard_type"])
    
    sector_columns = ["sector","subsector","asset_layer","exposure_unit","damage_cost_unit"]
    service_targets = [10,20,30,40,50,60,70,80,90,100]
    for asset_info in asset_data_details.itertuples():
        asset_id = asset_info.asset_id_column
        asset_service_columns = str(asset_info.criticality_columns).split(",")
        if asset_info.service_disruption_level == "asset":         
            no_adapt_sector_loss_file = os.path.join(
                                        no_adaptation_results,
                                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_sector_damages_losses.csv"
                                        )
            no_adapt_asset_loss_file = os.path.join(
                                        no_adaptation_results,
                                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages_losses.csv"
                                        )
            network_effect = False

        else:
            no_adapt_asset_loss_file = os.path.join(
                                        no_adaptation_results,
                                        f"{country}_{asset_info.asset_gpkg}_asset_damages_losses.parquet"
                                        )
            network_effect = True
            if asset_info.asset_gpkg == "energy":
                no_adapt_sector_loss_file = os.path.join(
                                            no_adaptation_results,
                                            f"{country}_{asset_info.asset_gpkg}_sector_damages_losses.csv"
                                            )
            else:
                no_adapt_sector_loss_file = os.path.join(
                                            no_adaptation_results,
                                            f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_sector_damages_losses.csv"
                                            )
        if os.path.isfile(no_adapt_sector_loss_file) is True:
            # damage_results = pd.read_parquet(damage_file)
            no_adapt_sector_loss_df = pd.read_csv(no_adapt_sector_loss_file)
            if network_effect is False:
                no_adapt_asset_loss_df = pd.read_csv(no_adapt_asset_loss_file)
            else:
                no_adapt_asset_loss_df = pd.read_parquet(no_adapt_asset_loss_file)
            
            for hk in ["min","mean","max"]:
                if len(asset_service_columns) > 1:
                    slc = "service_loss_disrupted"
                    no_adapt_sector_loss_df[
                            f"{slc}_{hk}_percentage"
                            ] = 1.0/len(asset_service_columns)*no_adapt_sector_loss_df[
                                    [f"{sl}_disrupted_{hk}_percentage" for sl in asset_service_columns]
                                    ].sum(axis=1)
                    no_adapt_asset_loss_df[
                            f"{slc}_{hk}_percentage_cumsum"
                            ] = 1.0/len(asset_service_columns)*no_adapt_asset_loss_df[
                                    [f"{sl}_disrupted_{hk}_percentage_cumsum" for sl in asset_service_columns]
                                    ].sum(axis=1)
                else:
                    slc = f"{asset_service_columns[0]}_disrupted"
            adapt_targets_df = []
            for res in no_adapt_sector_loss_df.itertuples():
                res_df = []
                rest_df = []
                for hk in ["min","mean","max"]:
                    service_loss = round(getattr(res,f"{slc}_{hk}_percentage"),2) + 1
                    targets = [st for st in service_targets if st >= 100.0 - service_loss]
                    targets_df = []
                    rm_df = []
                    for target in targets:
                        if network_effect is False:
                            no_adapt_asset_loss_df[
                                        f"{slc}_{hk}_percentage_cumsum"
                                    ] = no_adapt_asset_loss_df[f"{slc}_{hk}_percentage_cumsum"].round(decimals=2)
                            asset_choice = no_adapt_asset_loss_df[
                                            (no_adapt_asset_loss_df.hazard == res.hazard
                                            ) & (no_adapt_asset_loss_df.epoch == res.epoch
                                            ) & (no_adapt_asset_loss_df.rcp == res.rcp
                                            ) & (no_adapt_asset_loss_df.rp == res.rp
                                            )] 
                            select_assets = asset_choice[
                                                asset_choice[
                                                f"{slc}_{hk}_percentage_cumsum"
                                                ] <= target - (100 - service_loss)]
                            remaining_assets = asset_choice[
                                                asset_choice[
                                                f"{slc}_{hk}_percentage_cumsum"
                                                ] > target - (100 - service_loss)]
                        else:
                            no_adapt_asset_loss_df[
                                f"{slc}_percentage_cumsum"
                                    ] = no_adapt_asset_loss_df[f"{slc}_percentage_cumsum"].round(decimals=2)
                            damaged_assets = os.path.join(output_data_path,
                                    "no_adaptation",
                                    f"direct_damages_{development_scenario}",
                                    f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages.csv")
                            if os.path.isfile(damaged_assets) is True:
                                damaged_assets = pd.read_csv(damaged_assets)
                                if asset_info.asset_gpkg == "energy":
                                    asset_choice = no_adapt_asset_loss_df[
                                                    (no_adapt_asset_loss_df.epoch == res.epoch
                                                    ) & (no_adapt_asset_loss_df[
                                                        f"{slc}_percentage_cumsum"
                                                        ] <= target - (100 - service_loss))]
                                    # get all damaged energy combinations
                                    assets = asset_choice[["node_path","edge_path"]].to_numpy().flatten() 
                                else:
                                    asset_choice = no_adapt_asset_loss_df[
                                                    no_adapt_asset_loss_df[
                                                        f"{slc}_percentage_cumsum"
                                                    ] <= target]
                                    # get all damaged road combinations
                                    assets = asset_choice[["edge_path"]].to_numpy().flatten()

                                if len(assets) > 0:
                                    assets = [list(x) for x in set(tuple(x) for x in assets if x is not np.nan)] 
                                    assets = list(set([item for sublist in assets for item in sublist])) 
                                    select_assets = damaged_assets[damaged_assets[asset_id].isin(assets)]
                                    select_assets = select_assets[
                                            (select_assets.hazard == res.hazard
                                            ) & (select_assets.epoch == res.epoch
                                            ) & (select_assets.rcp == res.rcp
                                            ) & (select_assets.rp == res.rp
                                            )] 
                                    remaining_assets = damaged_assets[~damaged_assets[asset_id].isin(assets)]
                                    remaining_assets = remaining_assets[
                                            (remaining_assets.hazard == res.hazard
                                            ) & (remaining_assets.epoch == res.epoch
                                            ) & (remaining_assets.rcp == res.rcp
                                            ) & (remaining_assets.rp == res.rp
                                            )] 
                                else:
                                    select_assets = []
                            else:
                                select_assets = []

                        if len(select_assets) > 0:
                            select_assets["service_resilience_target_percentage"] = target
                            select_assets["asset_fix"] = 1
                            targets_df.append(select_assets[
                                                [asset_id] + sector_columns + [
                                                "service_resilience_target_percentage",
                                                "asset_fix"
                                                ] + hazard_columns])
                            remaining_assets["service_resilience_target_percentage"] = target
                            remaining_assets["asset_fix"] = 0
                            rm_df.append(remaining_assets[
                                                [asset_id] + sector_columns + [
                                                "service_resilience_target_percentage", 
                                                "asset_fix"
                                                ] + hazard_columns])

                    
                    res_df = get_damage_exposure_values(res_df,targets_df,
                                        [asset_id] + sector_columns  + hazard_columns,
                                        hk,
                                        os.path.join(
                                        direct_damages_results,
                                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages.csv"
                                        ))
                    rest = get_damage_exposure_values(rest_df,rm_df,
                                        [asset_id] + sector_columns  + hazard_columns,
                                        hk,
                                        os.path.join(
                                            output_data_path,
                                            "no_adaptation",
                                            f"direct_damages_{development_scenario}",
                                            f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages.csv"
                                            ))

                # breakpoint()
                if len(res_df) > 0:
                    res_df = pd.concat(res_df, axis = 1)
                    res_df = res_df.reset_index()
                    # breakpoint()
                    # print (res_df)
                    adapt_targets_df.append(res_df)
                if len(rest_df) > 0:
                    rest_df = pd.concat(rest_df, axis = 1)
                    rest_df = rest_df.reset_index()
                    # breakpoint()
                    # print (res_df)
                    adapt_targets_df.append(rest_df)

            if len(adapt_targets_df):
                adapt_targets_df = pd.concat(adapt_targets_df,axis=0,ignore_index=True)
                """Add costs
                """
                asset_df = gpd.read_file(os.path.join(
                                        processed_data_path,
                                        "infrastructure",
                                        asset_info.sector,
                                        f"{country}_{asset_info.asset_gpkg}.gpkg"),
                                    layer=asset_info.asset_layer)

                no_adapt_damage_df = pd.read_csv(os.path.join(
                                            output_data_path,
                                            "no_adaptation",
                                            f"direct_damages_{development_scenario}",
                                            f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages.csv"
                                            ))

                epochs = list(set(no_adapt_damage_df.epoch.values.tolist()))
                asset_costs_df = []
                for epoch in epochs:
                    asset_df = add_costs(asset_df,country,
                                    asset_info.sector,asset_info.asset_gpkg,
                                    asset_info.asset_layer,["construction"],epoch,development_scenario)
                    asset_df["epoch"] = epoch
                    # print (asset_df)
                    cost_df = pd.merge(asset_df,reduction_df,how="left",
                                    left_on=[asset_info.hazard_asset_damage_lookup_column],
                                    right_on=["asset_name"]).fillna(0)
                    cost_df = add_adaptation_costs(cost_df,asset_info.asset_layer,"construction_cost","cost_uplift")
                    asset_costs_df.append(
                            cost_df[
                                [asset_id,"hazard","epoch"] + [f"adaptation_investment_{hk}" for hk in ["min","mean","max"]]
                                ])

                asset_costs_df = pd.concat(asset_costs_df,axis=0,ignore_index=True)
                no_adapt_damage_df = pd.merge(no_adapt_damage_df,asset_costs_df,how="left",on=[asset_id,"hazard","epoch"])

                for hk in ["min","mean","max"]:
                    no_adapt_damage_df[f"adaptation_investment_{hk}"] = no_adapt_damage_df[f"exposure_{hk}"]*no_adapt_damage_df[f"adaptation_investment_{hk}"]


                adapt_targets_df = pd.merge(adapt_targets_df,
                                        no_adapt_damage_df[
                                            [asset_id] + sector_columns  + hazard_columns + [
                                            f"adaptation_investment_{hk}" for hk in ["min","mean","max"]
                                            ]],
                                        how="left",on=[asset_id] + sector_columns  + hazard_columns)
                adapt_targets_df.to_csv(os.path.join(direct_damages_results,
                                f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_targets_costs.csv"),index=False)


if __name__ == "__main__":
    CONFIG = load_config()
    try:
        country =  str(sys.argv[1])
        hazard_names = ast.literal_eval(str(sys.argv[2]))
        direct_damages_folder = str(sys.argv[3])
        damages_service_folder = str(sys.argv[4])
        network_csv = str(sys.argv[5])
        development_scenario = str(sys.argv[6])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()

    main(CONFIG,country,hazard_names,direct_damages_folder,
        damages_service_folder,
        network_csv,
        development_scenario)
