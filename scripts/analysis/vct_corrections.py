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
def main(config):

    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    baseline_year = 2023
    country = "vct"
    # edge_id sector  subsector   asset_layer exposure_unit   damage_cost_unit    hazard  isoa3   epoch   rcp rp
    development_scenarios = ["bau","sdg"]
    output_file = os.path.join(output_data_path,
                                "adaptation_outcomes",
                                "vct_energy_risks_investments_mod.xlsx")
    if os.path.isfile(output_file) is True:
        writer = pd.ExcelWriter(output_file,mode='a',if_sheet_exists='replace')
    else:
        writer = pd.ExcelWriter(output_file)
    for development_scenario in development_scenarios:
        direct_damages_results = os.path.join(output_data_path,
                                "with_adaptation",
                            f"direct_damages_{development_scenario}")
        for asset in ["areas"]:
            cost_file = os.path.join(
                                        direct_damages_results,
                                        f"{country}_energy_{asset}_asset_targets_costs.parquet"
                                    )
            cost_df = pd.read_parquet(cost_file)
            check_epoch = cost_df[(cost_df["hazard"] == "coastal") & (cost_df["rcp"] == "ssp126")]
            if len(check_epoch.index)  == 0:
                coastal = cost_df[(cost_df["hazard"] == "coastal") & (cost_df["rcp"] == "baseline")]
                if len(coastal.index) > 0:
                    coasts = []
                    for epoch in [2030,2050]:
                        l = coastal.copy()
                        l["epoch"] = epoch
                        l["rcp"] = "ssp126"
                        coasts.append(l)
                    coasts = pd.concat(coasts,axis=0,ignore_index=True)
                    cost_df = pd.concat([cost_df,coasts],axis=0,ignore_index=True)
                    del coastal, l, coasts
                cost_df.to_parquet(cost_file,index=False)
            else:
                print ("SSP126 for coastal already exists")

        service_loss_results = os.path.join(output_data_path,
                                "adaptation_outcomes",
                                "vct_energy_risks_investments.xlsx")
        service_df = pd.read_excel(service_loss_results,sheet_name=development_scenario)
        print (service_df) 

        service_flow_df = pd.read_parquet(
                            os.path.join(
                                output_data_path,
                                "no_adaptation",
                                f"damage_service_losses_{development_scenario}",
                                "vct_energy_asset_damages_losses.parquet"))
        pop_df = service_flow_df.drop_duplicates(subset=["destination_id","epoch"], keep="first")
        capacity_df = service_flow_df.drop_duplicates(subset=["origin_id","epoch"], keep="first")
        sum_dict = {'pop_2020':"sum"}
        pop_df = pop_df.groupby(
                    ["epoch","total_installed_capacity"]
                    ).agg(sum_dict).reset_index()
        print (pop_df)
        sum_dict = {'capacity_mw':"sum",'modified_capacity':"sum"}
        capacity_df = capacity_df.groupby(
                    ["epoch","total_installed_capacity"]
                    ).agg(sum_dict).reset_index()
        print (capacity_df)
        grid_df = pd.merge(capacity_df,pop_df,how="left",on=["epoch","total_installed_capacity"])
        grid_df["grid_demand"] = grid_df["pop_2020"]*grid_df["modified_capacity"]/grid_df["total_installed_capacity"]
        print (grid_df)
        grid_df["pop_factor"] = grid_df["pop_2020"] - grid_df["grid_demand"]

        service_df = pd.merge(service_df,
                        grid_df[["epoch","pop_2020","pop_factor"]],
                        how="left",on=["epoch"])

        service_df["customer_changes"] = np.where(service_df["customers_disrupted_percentage"].isin([0,100]),0,1)
        service_df["customers_disrupted"] = np.where(
                                service_df["customer_changes"] == 1,
                                service_df["customers_disrupted"] - service_df["pop_factor"],
                                service_df["customers_disrupted"])
        service_df["customers_disrupted_percentage"] = round(100.0*service_df["customers_disrupted"]/service_df["pop_2020"],1)
        service_df["service_resilience_achieved_percentage"] = 100.0 - service_df["customers_disrupted_percentage"]
        service_df.drop(["pop_2020","pop_factor","customer_changes"],axis=1,inplace=True)
        # service_df.to_csv(f"result_{development_scenario}.csv",index=False)
        service_df.to_excel(writer,sheet_name=development_scenario,index=False)
    writer.close()




if __name__ == "__main__":
    CONFIG = load_config()

    main(CONFIG)
