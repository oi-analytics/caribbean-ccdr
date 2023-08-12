"""Estimate direct damages to physical assets exposed to hazards

"""
import sys
import os

import pandas as pd
import geopandas as gpd
import itertools
import fiona
import numpy as np
import ast
import warnings
from add_costs import add_costs, add_installed_capacity_costs
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from analysis_utils import *
from tqdm import tqdm
tqdm.pandas()

def main(config):

    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']

    network_csv = os.path.join(processed_data_path,
                            "data_layers",
                            "network_layers_hazard_intersections_details.csv")
    countries = ["lca","dma","grd"]
    epochs = [2023,2030,2050]
    development_scenarios = ["bau","sdg"]
    asset_data_details = pd.read_csv(network_csv)[["sector","asset_gpkg"]]
    asset_data_details = asset_data_details.drop_duplicates(subset=["sector","asset_gpkg"],keep="first")
    for country in countries:
        output_file = os.path.join(processed_data_path,
                            "costs_and_options",
                            "asset_capital_costs.xlsx")
        if os.path.isfile(output_file) is True:
            writer = pd.ExcelWriter(output_file,mode='a',if_sheet_exists='replace')
        else:
            writer = pd.ExcelWriter(output_file)
        capital_costs = []
        for asset_info in asset_data_details.itertuples():
            asset_file = os.path.join(processed_data_path,
                                    "infrastructure",
                                    asset_info.sector,
                                    f"{country}_{asset_info.asset_gpkg}.gpkg")
            if os.path.isfile(asset_file) is True:
                layers = fiona.listlayers(asset_file)
                for asset_layer in layers:
                    asset_df = gpd.read_file(os.path.join(processed_data_path,
                                                        "infrastructure",
                                                        asset_info.sector,
                                                        f"{country}_{asset_info.asset_gpkg}.gpkg"),
                                                layer=asset_layer)
                    if asset_layer == 'edges':
                            asset_df['dimension'] = asset_df.geometry.length
                    elif asset_layer == 'areas':
                        asset_df['dimension'] = asset_df.geometry.area
                    else:
                        asset_df['dimension'] = 1
                    for development_scenario in development_scenarios:
                        for epoch in epochs:
                            if "capacity_mw" in asset_df.columns.values.tolist():
                                asset_df = asset_df[asset_df["capacity_mw"] == 0]
                                capacity_costs = add_installed_capacity_costs(country,
                                                                asset_info.sector,asset_info.asset_gpkg,
                                                                ["rehabilitation"],epoch,
                                                                development_scenario)
                                capacity_cost_min = capacity_costs["rehabilitation_cost_min"].sum()
                                capacity_cost_max = capacity_costs["rehabilitation_cost_max"].sum()
                            else:
                                capacity_cost_min = 0
                                capacity_cost_max = 0

                            if len(asset_df) > 0:
                                asset_df = add_costs(asset_df,country,
                                                    asset_info.sector,asset_info.asset_gpkg,
                                                    asset_layer,
                                                    ["rehabilitation"],epoch,
                                                    development_scenario)
                                capital_cost_min = capacity_cost_min + sum(asset_df["rehabilitation_cost_min"]*asset_df["dimension"])
                                capital_cost_max = capacity_cost_max + sum(asset_df["rehabilitation_cost_max"]*asset_df["dimension"])
                            
                            capital_cost_mean = 0.5*(capital_cost_min + capital_cost_max)

                            capital_costs.append({"sector":asset_info.sector,
                                                "subsector":asset_info.asset_gpkg,
                                                "asset_layer":asset_layer,
                                                "development_scenario":development_scenario,
                                                "epoch":epoch,
                                                "capital_cost_min":capital_cost_min,
                                                "capital_cost_mean":capital_cost_mean,
                                                "capital_cost_max":capital_cost_max})

        if len(capacity_costs) > 0:
            capital_costs = pd.DataFrame(capital_costs)
            sum_dict = dict([(f"capital_cost_{m}","sum") for m in ["min","mean","max"]])
            capital_costs = capital_costs.groupby(["sector","subsector","development_scenario","epoch"],
                                    dropna=False).agg(sum_dict).reset_index()


            capital_costs.to_excel(writer,sheet_name=country,index=False)
            writer.close()


if __name__ == "__main__":
    CONFIG = load_config()
    main(CONFIG)
