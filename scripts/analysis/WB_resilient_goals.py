"""Estimate direct damages to physical assets exposed to hazards

"""
import sys
import os

import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import interpolate
import warnings
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

def get_target_matches(dataframe,index_columns,
                    investment_column,
                    new_investment_column,
                    damage_column,
                    avoided_damage_column,
                    avoided_damage_target_column,
                    resilience_target_column,
                    avoided_damage_target,service_target):

    dataframe = dataframe.set_index(index_columns)
    index_values = list(set(dataframe.index.values.tolist()))
    t_df = []
    for idx in index_values:
        df = dataframe[dataframe.index == idx]
        damage = max(df[damage_column])
        new_investment = max(df[new_investment_column])
        investments = df[investment_column].values
        avoided_damages = df[avoided_damage_target_column].values
        resilience_values = df[resilience_target_column].values
        
        f_dam_inv = interpolate.interp1d(avoided_damages,investments,fill_value='extrapolate')
        f_res_inv = interpolate.interp1d(resilience_values,investments,fill_value='extrapolate')
        
        f_inv_dam = interpolate.interp1d(investments,avoided_damages,fill_value='extrapolate')
        f_inv_res = interpolate.interp1d(investments,resilience_values,fill_value='extrapolate')

        if avoided_damage_target is not None and service_target is not None:
            inv = max(f_dam_inv(avoided_damage_target),f_res_inv(service_target))
        elif avoided_damage_target is not None:
            inv = f_dam_inv(avoided_damage_target)
        elif service_target is not None:
            inv = f_res_inv(service_target)
        else:
            inv = max(investments)
        
        res_t = min(f_inv_res(inv),100.0)
        dam_t = min(f_inv_dam(inv),100.0)
        dam_v = (1 - 0.01*dam_t)*damage
        avd_dam_v = 0.01*dam_t*damage
        t_df.append(tuple(list(idx) + [inv,new_investment,dam_v,avd_dam_v,dam_t,res_t]))
        
    return pd.DataFrame(t_df,
                columns = index_columns + [investment_column,
                    new_investment_column,
                    damage_column,avoided_damage_column,
                    avoided_damage_target_column,resilience_target_column])

# def get_gdp_target_matches(dataframe,
#                     index_columns,
#                     investment_column,
#                     new_investment_column,
#                     damage_column,
#                     avoided_damage_column,
#                     damage_percentage_column,
#                     damage_percentage_target):

#     dataframe = dataframe.set_index(index_columns)
#     index_values = list(set(dataframe.index.values.tolist()))
#     t_df = []
#     for idx in index_values:
#         df = dataframe[dataframe.index == idx]
#         new_investment = max(df[new_investment_column])
#         investments = df[investment_column].values
#         damages_percentages = df[damage_percentage_column].values
#         if max(damages_percentages) > damage_percentage_target:
#             f_dam_inv = interpolate.interp1d(damages_percentages,investments,fill_value='extrapolate')
#             inv = f_dam_inv(damage_percentage_target)
#             t_df.append(list(idx) + [inv,new_investment,damage_percentage_target])
#         else:
#             t_df.append(list(idx) + [min(investments),max(damages)])
#     return pd.DataFrame(t_df,
#                 columns = index_columns + [
#                     investment_column,
#                     damage_column])

def esitmate_capital_stock_values(all_estimates):
    capital_stock_columns = [
                                "adaptation_investment",
                                "new_build_resilience_investment",
                                "total_investment",
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
    return all_estimates, cp_st_columns, stock_columns

def main(config,country,
        development_scenario,fixed_resilience):
    
    timeline = 2050.0 - 2023.0
    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    combined_results = os.path.join(output_data_path,
                                f"adaptation_outcomes","aggregated_results")
    index_columns  = ["sector","subsector",
                        "damage_cost_unit","isoa3",
                        "development_scenario",
                        "epoch","rcp","rp"]
    results_df = pd.read_excel(os.path.join(combined_results,
                                f"{country}_all_sectors_risks_investments.xlsx"),
                    sheet_name=development_scenario)
    results_df["service_resilience_achieved_percentage"] = np.where(results_df["existing_asset_adaptation_implemented"] == "yes",100,0)
    # results_df = results_df[results_df["rp"] > 1]
    # combined_results_df = pd.read_excel(os.path.join(combined_results,
    #                             f"{country}_total_risks_investments.xlsx"),
    #                 sheet_name=development_scenario)
    
    subsectors = list(set(results_df.subsector.values.tolist()))    
    goals_df = pd.read_excel(os.path.join(processed_data_path,
                                "data_layers",
                                f"service_targets_sdgs.xlsx"),
                    sheet_name="chosen_resilience_goals")

    goals_df = goals_df[goals_df["include"] == "Yes"]
    if fixed_resilience > 0:
        goals_df["resilience_target_percentage"] = fixed_resilience
    
    # subsectors = list(set(goals_df.subsector.values.tolist()))
    target_df = []
    for subsector in subsectors:
        g_df = goals_df[goals_df["subsector"] == subsector]
        sub_df = results_df[results_df["subsector"] == subsector]
        if len(g_df[g_df["damage_effect"] == "yes"]) > 0:
            avoided_damage_target = max(g_df[g_df["damage_effect"] == "yes"]["resilience_target_percentage"])
        else:
            avoided_damage_target = None
        if len(g_df[g_df["service_effect"] == "yes"]) > 0:
            service_target = max(g_df[g_df["service_effect"] == "yes"]["resilience_target_percentage"])
        else:
            service_target = None
        dfs = []
        for hk in ["min","mean","max"]: 
            df = get_target_matches(sub_df,index_columns,
                                    f"adaptation_investment_{hk}",
                                    f"new_build_resilience_investment_{hk}",
                                    f"damage_{hk}",
                                    f"avoided_damage_{hk}",
                                    f"avoided_damage_{hk}_percentage",
                                    "service_resilience_achieved_percentage",
                                    avoided_damage_target,service_target)
            dfs.append(df.set_index(index_columns))
            

        dfs = pd.concat(dfs,axis=1)
        dfs = dfs.loc[:,~dfs.columns.duplicated()].copy()
        target_df.append(dfs.reset_index())

    target_df = pd.concat(target_df,axis=0,ignore_index=True)
    for hk in ["min","mean","max"]:
        target_df[
                f"total_investment_{hk}"
                ] = target_df[
                        f"adaptation_investment_{hk}"
                        ] + target_df[f"new_build_resilience_investment_{hk}"]
    target_df["development_scenario"] = development_scenario
 
    selected_columns = ["sector","subsector",
                        "damage_cost_unit",
                        "epoch","rcp","rp","isoa3",
                        "development_scenario",
                        "adaptation_investment_min",
                        "adaptation_investment_mean",
                        "adaptation_investment_max",
                        "new_build_resilience_investment_min",
                        "new_build_resilience_investment_mean",
                        "new_build_resilience_investment_max",
                        "total_investment_min",
                        "total_investment_mean",
                        "total_investment_max",
                        "damage_min","damage_mean","damage_max",
                        "service_resilience_achieved_percentage",
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
    all_estimates = pd.merge(target_df[selected_columns],
                              capital_costs,
                              how="left",
                              on=["development_scenario","epoch"])
    
    all_estimates, cp_st_columns, stock_columns = esitmate_capital_stock_values(all_estimates)
    
    if fixed_resilience == 0:
        output_file = os.path.join(combined_results,
                                f"{country}_all_sectors_risks_investments_with_resilience_goals.xlsx")
    else:
        output_file = os.path.join(combined_results,
                                f"{country}_all_sectors_risks_investments_with_{fixed_resilience}_percent_resilience.xlsx")
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)

    all_estimates.to_excel(writer,sheet_name=development_scenario,index=False)
    writer.close()
    del results_df
    
    """Get tthe combined resilience goal across all sectors
    """
    index_columns  = ["damage_cost_unit","isoa3",
                        "development_scenario",
                        "epoch","rcp","rp"]
    combined_results_df = all_estimates.groupby(index_columns)[cp_st_columns].sum().reset_index()
    # gdp_target = goals_df[goals_df["subsector"] == "all"]["resilience_target_percentage"].values[0]
    gdp_estimates = pd.read_excel(os.path.join(processed_data_path,
                                "data_layers",
                                "gdp_population_estimates.xlsx"),
                                sheet_name="gdp_pop")
    gdp_estimates = gdp_estimates[gdp_estimates["iso_code"] == country]
    gdp_estimates["gdp"] = 1.0e9*gdp_estimates["gdp"]
    combined_results_df = pd.merge(combined_results_df,
                        gdp_estimates[["iso_code","epoch","gdp"]],
                        how="left",left_on=["isoa3","epoch"], right_on=["iso_code","epoch"])
    for hk in ["min","mean","max"]:
        combined_results_df[
                f"total_investment_{hk}_as_gdp_percentage_per_year"
                ] = 100.0*combined_results_df[f"total_investment_{hk}"]/(timeline*combined_results_df["gdp"])
    for hk in ["min","mean","max"]:
        combined_results_df[
                f"damage_{hk}_as_gdp_percentage"
                ] = 100.0*combined_results_df[f"damage_{hk}"]/combined_results_df["gdp"]
    for hk in ["min","mean","max"]:
        combined_results_df[
                f"avoided_damage_{hk}_as_gdp_percentage"
                ] = 100.0*combined_results_df[f"avoided_damage_{hk}"]/combined_results_df["gdp"]
    
    combined_results_df.drop(["iso_code","gdp"],axis=1,inplace=True)

    combined_results_df = pd.merge(combined_results_df,
                              capital_costs,
                              how="left",
                              on=["development_scenario","epoch"])
    totals, cp_st_columns, stock_columns = esitmate_capital_stock_values(combined_results_df)

    if fixed_resilience == 0:
        output_file = os.path.join(combined_results,
                                f"{country}_total_risks_investments_with_resilience_goals.xlsx")
    else:
        output_file = os.path.join(combined_results,
                                f"{country}_total_risks_investments_with_{fixed_resilience}_percent_resilience.xlsx")
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)

    totals.to_excel(writer,sheet_name=development_scenario,index=False)
    writer.close()

    sum_columns = cp_st_columns + stock_columns + [f"damage_{hk}_as_gdp_percentage" for hk in ["min","mean","max"]]
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
    if fixed_resilience == 0:
        output_file = os.path.join(combined_results,
                                f"{country}_total_risks_investments_time_series_with_resilience_goals.xlsx")
    else:
        output_file = os.path.join(combined_results,
                            f"{country}_total_risks_investments_time_series_with_{fixed_resilience}_percent_resilience.xlsx")
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
        fixed_resilience = int(sys.argv[3])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()

    main(CONFIG,country,
        development_scenario,fixed_resilience)
