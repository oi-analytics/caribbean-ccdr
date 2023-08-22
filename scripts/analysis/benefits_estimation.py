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

def main(config,country,
        network_csv,
        development_scenario):

    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    baseline_year = 2023
    no_adaptation_results = os.path.join(output_data_path,
                                "no_adaptation",
                                f"damage_service_losses_{development_scenario}")
    with_adaptation_results = os.path.join(output_data_path,
                                "with_adaptation",
                                f"damage_service_losses_{development_scenario}")

    with_adaptation_costs = os.path.join(output_data_path,
                                "with_adaptation",
                                f"direct_damages_{development_scenario}")

    combined_results = os.path.join(output_data_path,
                                f"adaptation_outcomes")
    if os.path.exists(combined_results) == False:
        os.mkdir(combined_results)

    asset_data_details = pd.read_csv(network_csv)
    asset_data_details = asset_data_details.drop_duplicates(subset=["asset_gpkg"],keep="first")
    for asset_info in asset_data_details.itertuples():
        asset_service_columns = str(asset_info.criticality_columns).split(",")
        # output_file = os.path.join(combined_results,
        #                     f"{country}_{asset_info.asset_gpkg}_risks_investments.xlsx")
        # if os.path.isfile(output_file) is True:
        #     writer = pd.ExcelWriter(output_file,mode='a',if_sheet_exists='replace')
        # else:
        #     writer = pd.ExcelWriter(output_file)
        if asset_info.asset_gpkg != "energy" or country == "grd":    
            no_adapt_sector_loss_file = os.path.join(
                                        no_adaptation_results,
                                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_sector_damages_losses.csv"
                                        )
            with_adapt_sector_loss_file = os.path.join(
                                        with_adaptation_results,
                                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_sector_damages_losses.csv"
                                        )
            with_adapt_asset_costs_file = os.path.join(with_adaptation_costs,
                                f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_targets_costs.parquet")
        else:
            no_adapt_sector_loss_file = os.path.join(
                                        no_adaptation_results,
                                        f"{country}_{asset_info.asset_gpkg}_sector_damages_losses.csv"
                                        )
            with_adapt_sector_loss_file = os.path.join(
                                        with_adaptation_results,
                                        f"{country}_{asset_info.asset_gpkg}_sector_damages_losses.csv"
                                        )

        if os.path.isfile(no_adapt_sector_loss_file) is True:
            output_file = os.path.join(combined_results,
                                f"{country}_{asset_info.asset_gpkg}_risks_investments.xlsx")
            if os.path.isfile(output_file) is True:
                writer = pd.ExcelWriter(output_file,mode='a',if_sheet_exists='replace')
            else:
                writer = pd.ExcelWriter(output_file)
            no_adapt_sector_loss_df = pd.read_csv(no_adapt_sector_loss_file)
            with_adapt_sector_loss_df = pd.read_csv(with_adapt_sector_loss_file)
            if asset_info.asset_gpkg != "energy" or country == "grd":
                with_adapt_asset_costs_file = os.path.join(with_adaptation_costs,
                                f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_targets_costs.parquet")
                with_adapt_asset_costs_df = pd.read_parquet(with_adapt_asset_costs_file)
            else:
                layers = [os.path.join(with_adaptation_costs,
                                        f"{country}_energy_{asset_layer}_asset_targets_costs.parquet"
                                ) for asset_layer in ["nodes","edges","areas"]]
                with_adapt_asset_costs_df = [
                            pd.read_parquet(layer) for layer in layers if os.path.isfile(layer) is True
                            ]
                with_adapt_asset_costs_df = pd.concat(with_adapt_asset_costs_df,axis=0,ignore_index=True)




            loss_columns = [c for c in no_adapt_sector_loss_df.columns.values.tolist() if "_min" in c or "_mean" in c or "_max" in c]
            index_columns = [c for c in no_adapt_sector_loss_df.columns.values.tolist() if c not in loss_columns]
            # damage_columns = [c for c in no_adapt_sector_loss_df.columns.values.toist() if "exposure_" in c or "damage_" in c]
            # service_columns = [c for c in loss_columns if c not in damage_columns]

            # print (with_adapt_sector_loss_df.columns.values.tolist())
            # print (no_adapt_sector_loss_df.columns.values.tolist())
            # print (index_columns)
            with_adapt_asset_costs_df = with_adapt_asset_costs_df[with_adapt_asset_costs_df["asset_fix"] == 1]
            sum_dict = dict([
                            (f"adaptation_investment_{hk}","sum") for hk in ["min","mean","max"]
                            ])
            idx_columns = [c for c in with_adapt_asset_costs_df.columns.values.tolist() if c in index_columns]
            with_adapt_asset_costs_df = with_adapt_asset_costs_df.groupby(
                                idx_columns + [
                                "service_resilience_target_percentage"],
                                dropna=False).agg(sum_dict).reset_index()

            with_adapt_sector_loss_df = pd.merge(with_adapt_sector_loss_df,
                                            with_adapt_asset_costs_df,
                                            how="left",
                                            on=idx_columns + [
                                                "service_resilience_target_percentage"])
            with_adapt_sector_loss_df.drop(["service_resilience_target_percentage"],axis=1,inplace=True)

            with_adapt_sector_loss_df["existing_asset_adaptation_implemented"] = "yes"
            no_adapt_sector_loss_df["existing_asset_adaptation_implemented"] = "no"

            with_adapt_sector_loss_df = pd.concat([no_adapt_sector_loss_df,
                                            with_adapt_sector_loss_df],
                                            axis=0,ignore_index=True)
            no_adapt_sector_loss_df.drop("existing_asset_adaptation_implemented",axis=1,inplace=True)
            for hk in ["min","mean","max"]:
                with_adapt_sector_loss_df[
                        f"adaptation_investment_{hk}"
                        ] = with_adapt_sector_loss_df[f"adaptation_investment_{hk}"].fillna(0)

            for hk in ["min","mean","max"]:
                if len(asset_service_columns) > 1:
                    slc = "weighted_service_disrupted"
                    no_adapt_sector_loss_df[
                            f"{slc}_{hk}_percentage"
                            ] = 1.0/len(asset_service_columns)*no_adapt_sector_loss_df[
                                    [f"{sl}_disrupted_{hk}_percentage" for sl in asset_service_columns]
                                    ].sum(axis=1)
                    with_adapt_sector_loss_df[
                            f"{slc}_{hk}_percentage"
                            ] = 1.0/len(asset_service_columns)*with_adapt_sector_loss_df[
                                    [f"{sl}_disrupted_{hk}_percentage" for sl in asset_service_columns]
                                    ].sum(axis=1)
                else:
                    slc = f"{asset_service_columns[0]}_disrupted"

                no_adapt_sector_loss_df[
                        f"service_resilience_achieved_{hk}_percentage"
                        ] = 100.0 - no_adapt_sector_loss_df[f"{slc}_{hk}_percentage"]
                with_adapt_sector_loss_df[
                        f"service_resilience_achieved_{hk}_percentage"
                        ] = 100.0 - with_adapt_sector_loss_df[f"{slc}_{hk}_percentage"]

            loss_columns = [c for c in no_adapt_sector_loss_df.columns.values.tolist() if "_min" in c or "_mean" in c or "_max" in c]
            rename_dict = dict([
                            (f"{c}",f"no_adaptation_{c}") for c in loss_columns
                            ])
            no_adapt_sector_loss_df.rename(columns=rename_dict,inplace=True)
            rename_dict = dict([
                            (f"{c}",f"with_adaptation_{c}") for c in loss_columns
                            ])
            with_adapt_sector_loss_df.rename(columns=rename_dict,inplace=True)

            # print (with_adapt_sector_loss_df.columns.values.tolist())
            # print (no_adapt_sector_loss_df.columns.values.tolist())
            # print (index_columns)
            with_adapt_sector_loss_df = pd.merge(with_adapt_sector_loss_df,
                                            no_adapt_sector_loss_df,
                                            how="left",on=index_columns)
            # print (with_adapt_sector_loss_df)

            with_adapt_sector_loss_df[
                        loss_columns
                        ] = with_adapt_sector_loss_df[
                                [f"no_adaptation_{c}" for c in loss_columns]
                                ] -  with_adapt_sector_loss_df[
                                        [f"with_adaptation_{c}" for c in loss_columns]
                                        ].values

            exposure_damage_columns = [c for c in loss_columns if "exposure_" in c or "damage_" in c]
            rename_dict = dict([
                            (f"{c}",f"avoided_{c}") for c in exposure_damage_columns
                            ])
            with_adapt_sector_loss_df.rename(columns=rename_dict,inplace=True)

            damage_columns = [c for c in exposure_damage_columns if "damage_" in c]
            for d in damage_columns:
                with_adapt_sector_loss_df[
                    f"avoided_{d}_percentage"
                    ] = 100.0*with_adapt_sector_loss_df[f"avoided_{d}"]/with_adapt_sector_loss_df[f"no_adaptation_{d}"]

            with_adapt_sector_loss_df.drop(
                    [f"no_adaptation_{c}" for c in loss_columns
                    ],
                    axis=1,inplace=True)
            service_loss_columns = [c for c in loss_columns if c not in exposure_damage_columns]
            with_adapt_sector_loss_df.drop(service_loss_columns,axis=1,inplace=True)
            rename_dict = dict([
                            (f"with_adaptation_{c}",f"{c}") for c in loss_columns
                            ])
            with_adapt_sector_loss_df.rename(columns=rename_dict,inplace=True)

            loss_columns = [c for c in with_adapt_sector_loss_df.columns.values.tolist() if "_min" in c or "_mean" in c or "_max" in c]
            index_columns = [c for c in with_adapt_sector_loss_df.columns.values.tolist() if c not in loss_columns]

            unique_loss_columns = list(set([c.replace("_min","").replace("_mean","").replace("_max","") for c in loss_columns]))
            for uc in unique_loss_columns:
                if f"{uc}_min" in loss_columns:
                    min_col = f"{uc}_min"
                    max_col = f"{uc}_max"
                    del_cols = [f"{uc}_min",f"{uc}_mean"]
                else:
                    min_col = f"{uc.replace('_percentage','')}_min_percentage"
                    max_col = f"{uc.replace('_percentage','')}_max_percentage"
                    del_cols = [f"{uc.replace('_percentage','')}_min_percentage",f"{uc.replace('_percentage','')}_mean_percentage"]
                if with_adapt_sector_loss_df[min_col].equals(with_adapt_sector_loss_df[max_col]):
                    with_adapt_sector_loss_df.rename(columns={max_col:uc},inplace=True)
                    with_adapt_sector_loss_df.drop(del_cols,axis=1,inplace=True)
                    with_adapt_sector_loss_df[uc] = with_adapt_sector_loss_df[uc].round(1)

            with_adapt_sector_loss_df["development_scenario"] = development_scenario
            index_columns += ["development_scenario"]
            if asset_info.asset_gpkg == "energy":
                reduction_df = get_adaptation_uplifts_reductions()
                hazard_mapping = pd.read_csv(os.path.join(processed_data_path,
                                        "damage_curves",
                                        "hazard_damage_parameters.csv"))[["hazard","hazard_type"]]
                reduction_df = pd.merge(hazard_mapping,reduction_df,how="left",on=["hazard_type"])
                reduction_df = reduction_df[reduction_df["subsector"] == "energy"]
                del hazard_mapping
                installed_capacities = pd.read_excel(os.path.join(
                                processed_data_path,
                                "data_layers",
                                "service_targets_sdgs.xlsx"),
                                sheet_name="sdg_energy_supply")
                installed_capacities = installed_capacities[
                                    (installed_capacities["epoch"] > baseline_year
                                    ) & (installed_capacities["iso_code"] == country)
                                    ]
                installed_capacities["asset_type"] = installed_capacities["asset_type"].str.lower()
                installed_capacities = installed_capacities[installed_capacities["asset_type"] != "diesel"]
                costs = pd.read_excel(os.path.join(
                            processed_data_path,
                            "costs_and_options",
                            f"asset_construction_costs.xlsx"),
                            sheet_name="energy")
                installed_capacities = pd.merge(installed_capacities,costs,how="left")
                del costs
                installed_capacities[
                        "construction_cost_min"
                        ] = installed_capacities[
                                f"{development_scenario}_capacity_mw_new"
                                ]*installed_capacities["cost_min"]
                installed_capacities[
                        "construction_cost_max"
                        ] = installed_capacities[
                                f"{development_scenario}_capacity_mw_new"
                                ]*installed_capacities["cost_max"]
                installed_capacities[
                        "construction_cost_mean"
                        ] = 0.5*(
                                installed_capacities[
                                    "construction_cost_min"
                                    ] + installed_capacities[
                                        "construction_cost_max"]
                                )

                installed_capacities = pd.merge(installed_capacities
                                        ,reduction_df,how="left",
                                        left_on=["asset_type"],
                                        right_on=["asset_name"]).fillna(0)

                installed_capacities[
                        [f"new_build_resilience_investment_{hk}" for hk in ["min","mean","max"]]
                            ] = installed_capacities[
                                    [f"construction_cost_{hk}" for hk in ["min","mean","max"]]
                                ].multiply(installed_capacities["cost_uplift"],axis="index")
                installed_capacities = installed_capacities.groupby(
                                        ["hazard","epoch"]
                                        )[[f"new_build_resilience_investment_{hk}" for hk in ["min","mean","max"]]].sum().reset_index()
                with_adapt_sector_loss_df = pd.merge(with_adapt_sector_loss_df,
                                        installed_capacities,
                                        how="left",on=["hazard","epoch"]).fillna(0)


            investment_columns = [c for c in with_adapt_sector_loss_df.columns.values.tolist() if "_investment_" in c]
            remaining_columns = [c for c in with_adapt_sector_loss_df.columns.values.tolist() if c not in index_columns + investment_columns]
            # with_adapt_sector_loss_df[index_columns + investment_columns + remaining_columns].to_csv(
            #                 os.path.join(combined_results,
            #                 f"{country}_{asset_info.asset_gpkg}_adaptation_{development_scenario}.csv"),index=False)
            with_adapt_sector_loss_df = with_adapt_sector_loss_df.drop_duplicates(subset=
                index_columns + ["damage_max",
                "service_resilience_achieved_percentage"],keep="first")
            with_adapt_sector_loss_df[
                    index_columns + investment_columns + remaining_columns
                    ].to_excel(writer,sheet_name=development_scenario,index=False)
            writer.close()

if __name__ == "__main__":
    CONFIG = load_config()
    try:
        country =  str(sys.argv[1])
        network_csv = str(sys.argv[2])
        development_scenario = str(sys.argv[3])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()

    main(CONFIG,country,
        network_csv,
        development_scenario)
