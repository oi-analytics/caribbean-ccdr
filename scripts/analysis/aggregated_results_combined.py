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

def estimate_time_series(hazard_data_details,
                        index_cols,
                        value_col
                        ):
    
    years = sorted(list(set(hazard_data_details.epoch.values.tolist())))
    hazard_data_details.epoch = hazard_data_details.epoch.astype(int)
    
    start_year = years[0]
    end_year = years[-1]
    timeseries = np.arange(start_year,end_year+1,1)

    baseline_df = hazard_data_details[hazard_data_details.epoch == 2023]
    future_df = hazard_data_details[hazard_data_details.epoch != 2023]
    
    rcp_rp = list(
                        set(
                            zip(
                                future_df.rcp.values.tolist(),
                                future_df.rp.values.tolist()
                                )
                            )
                        )
    damages_time_series = []
    idx_cols = [c for c in index_cols if c != "epoch"]
    for ix, (rcp,rp) in enumerate(rcp_rp):
        f_df = future_df[(future_df.rcp == rcp) & (future_df.rp == rp)]
        b_df = baseline_df[baseline_df.rp == rp]
        b_df["rcp"] = rcp
        df = pd.concat([b_df,f_df],axis=0,ignore_index=True)
        years = sorted(list(set(df.epoch.values.tolist())))
        df = df[index_cols + [value_col]]
        df = (df.set_index(idx_cols).pivot(
                            columns="epoch"
                                )[value_col].reset_index().rename_axis(None, axis=1)).fillna(0)
        # df.columns = df.columns.astype(str)
        # df.columns = index_cols + [start_year] + years[1:]
        # df["hazard"] = haz
        # df["model"] = mod
        # df["rcp"] = rcp
        # years = [start_year] + years[1:]
        # years = sorted(list(set(df.epoch.values.tolist())))
        series = np.array([list(timeseries)*len(df.index)]).reshape(len(df.index),len(timeseries))
        df[series[0]] = interp1d(years,df[years],fill_value="extrapolate",bounds_error=False)(series[0])
        df[series[0]] = df[series[0]].clip(lower=0.0)
        df = df.melt(id_vars=idx_cols,var_name="epoch",value_name=value_col)
        df = df.sort_values("epoch")
        damages_time_series.append(df)
        

    damages_time_series = pd.concat(damages_time_series,axis=0,ignore_index=False)

    return damages_time_series

def main(config,country,
        development_scenario):

    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    combined_results = os.path.join(output_data_path,
                                f"adaptation_outcomes")
    selected_columns = ["sector","subsector",
                        "damage_cost_unit",
                        "epoch","rcp","rp","isoa3",
                        "existing_asset_adaptation_implemented",
                        "development_scenario",
                        "adaptation_investment_min",
                        "adaptation_investment_mean",
                        "adaptation_investment_max",
                        "new_build_resilience_investment_min",
                        "new_build_resilience_investment_mean",
                        "new_build_resilience_investment_max",
                        "damage_min","damage_mean","damage_max",
                        "avoided_damage_min","avoided_damage_mean","avoided_damage_max",
                        "avoided_damage_min_percentage","avoided_damage_mean_percentage",
                        "avoided_damage_max_percentage"]

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

    all_estimates = pd.read_excel(os.path.join(
                        combined_results,
                        f"{country}_combined_hazard_sector_risks_investments.xlsx"),
                    sheet_name=development_scenario)
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
                            f"{country}_all_sectors_risks_investments.xlsx")
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
                        "epoch","rcp","rp","isoa3",
                        "existing_asset_adaptation_implemented",
                        "development_scenario"
                       ]
    no_adapt_option = all_estimates[all_estimates["existing_asset_adaptation_implemented"] == "no"]
    no_adapt_option = no_adapt_option.groupby(index_columns)[sum_columns].sum().reset_index()

    full_adapt_option = all_estimates[
                            all_estimates["existing_asset_adaptation_implemented"] == "yes"
                            ]
    full_adapt_option = full_adapt_option.groupby(index_columns)[sum_columns].sum().reset_index()
    totals = pd.concat([no_adapt_option,full_adapt_option],axis=0,ignore_index=True)

    output_file = os.path.join(combined_results,
                            f"{country}_aggregated_risks_investments.xlsx")
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
                            f"{country}_aggregated_risks_investments_time_series.xlsx")
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
        development_scenario = str(sys.argv[2])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()

    main(CONFIG,country,
        development_scenario)
