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
        development_scenario):

    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    combined_results = os.path.join(output_data_path,
                                f"adaptation_outcomes")
    asset_data_details = pd.read_csv(network_csv)

    asset_data_details = asset_data_details.drop_duplicates(subset=["asset_gpkg"],keep="first")
    selected_columns = ["sector","subsector",
                        "damage_cost_unit",
                        "hazard","epoch","rcp","rp","isoa3",
                        "existing_asset_adaptation_implemented",
                        "development_scenario",
                        "adaptation_investment_min",
                        "adaptation_investment_mean",
                        "adaptation_investment_max",
                        "new_build_resilience_investment_min",
                        "new_build_resilience_investment_mean",
                        "new_build_resilience_investment_max",
                        "damage_min","damage_mean","damage_max",
                        "service_resilience_achieved_percentage",
                        "avoided_damage_min","avoided_damage_mean","avoided_damage_max",
                        "avoided_damage_min_percentage","avoided_damage_mean_percentage",
                        "avoided_damage_max_percentage"]

    # gdp_estimates = pd.read_excel(os.path.join(processed_data_path,
    #                             "data_layers",
    #                             "gdp_population_estimates.xlsx"),
    #                             sheet_name="gdp_pop")
    # gdp_estimates = gdp_estimates[gdp_estimates["iso_code"] == country]
    # gdp_estimates["gdp"] = 1.0e9*gdp_estimates["gdp"]

    capital_costs = pd.read_excel(os.path.join(processed_data_path,
                                "costs_and_options",
                                "asset_capital_costs.xlsx"),
                                sheet_name=country)
    capital_costs = capital_costs[capital_costs["development_scenario"] == development_scenario]
    capital_costs = capital_costs.groupby(
                            ["development_scenario",
                            "epoch"]
                            )[["capital_cost_min",
                                "capital_cost_mean",
                                "capital_cost_max"]].sum().reset_index()
    all_estimates = []
    for asset_info in asset_data_details.itertuples():
        input_file = os.path.join(combined_results,
                            f"{country}_{asset_info.asset_gpkg}_risks_investments.xlsx")
        if os.path.isfile(input_file) is True:
            df = pd.read_excel(
                        input_file,
                        sheet_name=development_scenario)
            columns = [c for c in df.columns.values.tolist() if c in selected_columns]
            all_estimates.append(df[columns])

    all_estimates = pd.concat(all_estimates,axis=0,ignore_index=True).fillna(0)
    # all_estimates = pd.merge(all_estimates,gdp_estimates[["epoch","gdp"]],how="left",on=["epoch"])
    all_estimates = pd.merge(all_estimates,
                              capital_costs,
                              how="left",
                              on=["development_scenario","epoch"])
    # for t in ["min","mean","max"]:
    #     all_estimates[f"capital_cost_{t}"] = capital_costs[f"capital_cost_{t}"].sum()

    # capital_stock_columns = [
    #                             "adaptation_investment_min",
    #                             "adaptation_investment_mean",
    #                             "adaptation_investment_max",
    #                             "new_build_resilience_investment_min",
    #                             "new_build_resilience_investment_mean",
    #                             "new_build_resilience_investment_max",
    #                             "damage_min","damage_mean","damage_max",
    #                             "avoided_damage_min","avoided_damage_mean","avoided_damage_max"
    #                         ]
    capital_stock_columns = [
                                "adaptation_investment",
                                "new_build_resilience_investment",
                                "damage",
                                "avoided_damage"
                            ]

    # gdp_columns = []
    # for col in capital_stock_columns:
    #     all_estimates[f"{col}_as_gdp_percentage"] = 100.0*all_estimates[col]/all_estimates["gdp"]
    #     gdp_columns.append(f"{col}_as_gdp_percentage")
    stock_columns = []
    cp_st_columns = []
    for col in capital_stock_columns:
        for col_type in ["min","mean","max"]:
            all_estimates[
                f"{col}_as_stock_percentage_{col_type}"
                ] = 100.0*all_estimates[f"{col}_{col_type}"]/all_estimates[f"capital_cost_{col_type}"]
            cp_st_columns.append(f"{col}_{col_type}")
            stock_columns.append(f"{col}_as_stock_percentage_{col_type}")

        all_estimates[[f'{col}_min',
                        f'{col}_mean',
                        f'{col}_max']]=np.sort(all_estimates[[f'{col}_min',
                                                            f'{col}_mean',
                                                            f'{col}_max']].values,axis=1)
        all_estimates[[f'{col}_as_stock_percentage_min',
                        f'{col}_as_stock_percentage_mean',
                        f'{col}_as_stock_percentage_max']]=np.sort(all_estimates[[f'{col}_as_stock_percentage_min',
                                                                    f'{col}_as_stock_percentage_mean',
                                                                    f'{col}_as_stock_percentage_max']].values,axis=1)


    all_estimates.drop(["capital_cost_min","capital_cost_mean","capital_cost_max"],axis=1,inplace=True)

    output_file = os.path.join(combined_results,
                            f"{country}_all_assets_risks_investments.xlsx")
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)

    all_estimates.to_excel(writer,sheet_name=development_scenario,index=False)
    writer.close()

    sum_columns = cp_st_columns + stock_columns
    # sum_columns = cp_st_columns
    index_columns = [
                        "damage_cost_unit",
                        "hazard","epoch","rcp","rp","isoa3",
                        "existing_asset_adaptation_implemented",
                        "development_scenario"
                       ]
    no_adapt_option = all_estimates[all_estimates["existing_asset_adaptation_implemented"] == "no"]
    no_adapt_option = no_adapt_option.groupby(index_columns)[sum_columns].sum().reset_index()

    full_adapt_option = all_estimates[(
                            all_estimates["existing_asset_adaptation_implemented"] == "yes"
                            ) & (
                            all_estimates["service_resilience_achieved_percentage"] == 100.0
                            )]
    full_adapt_option = full_adapt_option.groupby(index_columns)[sum_columns].sum().reset_index()
    totals = pd.concat([no_adapt_option,full_adapt_option],axis=0,ignore_index=True)

    output_file = os.path.join(combined_results,
                            f"{country}_total_risks_investments.xlsx")
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)

    totals.to_excel(writer,sheet_name=development_scenario,index=False)
    writer.close()

    totals_series = []
    for col in sum_columns:
        series_df = estimate_time_series(totals,
                        index_columns,
                        col
                        )
        if len(totals_series) > 0:
            totals_series = pd.merge(totals_series,series_df,how="left",on=index_columns)
        else:
            totals_series = series_df.copy()

    # totals_series = totals_series.reindex([c for c in index_columns if c != "epoch"])
    totals_series = totals_series.sort_values([c for c in index_columns if c != "epoch"],ascending=True)
    # totals_series = totals_series.sort_values(index_columns,ascending=True)
    output_file = os.path.join(combined_results,
                            f"{country}_total_risks_investments_time_series.xlsx")
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)

    totals_series.to_excel(writer,sheet_name=development_scenario,index=False)
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
