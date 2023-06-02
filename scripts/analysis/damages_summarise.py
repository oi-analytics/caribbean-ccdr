"""Estimate direct damages to physical assets exposed to hazards

"""
import sys
import os

import pandas as pd
import geopandas as gpd
import numpy as np
import ast
import warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from analysis_utils import *
from tqdm import tqdm
tqdm.pandas()

def quantiles(dataframe,grouping_by_columns,grouped_columns):
    grouped = dataframe.groupby(grouping_by_columns,dropna=False)[grouped_columns].agg([np.min, np.mean, np.max]).reset_index()
    grouped.columns = grouping_by_columns + [f"{prefix}_{agg_name}" for prefix, agg_name in grouped.columns if prefix not in grouping_by_columns]
    
    return grouped

def main(config,country,hazard_names,direct_damages_folder,
        summary_results_folder,
        network_csv,
        parameter_txt_file):

    processed_data_path = config['paths']['data']
    output_data_path = config['paths']['results']
    
    direct_damages_results = os.path.join(output_data_path,direct_damages_folder)

    summary_results = os.path.join(output_data_path,summary_results_folder)
    if os.path.exists(summary_results) == False:
        os.mkdir(summary_results)

    asset_data_details = pd.read_csv(network_csv)
    param_values = open(parameter_txt_file)
    param_values = [line.split(',')[0] for line in param_values.readlines()]
    
    
    # hazard_data_details = pd.read_csv(hazard_csv,encoding="latin1")
    # hazard_data_details.columns = hazard_data_details.columns.str.replace('-', '_')
    # hazard_cols = [c for c in hazard_data_details.columns.values.tolist() if c not in ["fname","key"]]
    uncertainty_columns = ['damage_uncertainty_parameter','cost_uncertainty_parameter']
    for asset_info in asset_data_details.itertuples():
        asset_id = asset_info.asset_id_column
        asset_damages_results = os.path.join(direct_damages_results,f"{asset_info.asset_gpkg}_{asset_info.asset_layer}")

        asset_damages = []
        sector_damages = []
        damage_sensitivity = []
        for hazard_name in hazard_names:
            # Process the exposure and damage results
            hazard_data_details = pd.read_csv(os.path.join(processed_data_path,
                                                        "hazards",
                                                        f"{hazard_name}_{country}.csv"),
                                                encoding="latin1")
            hazard_data_details.columns = hazard_data_details.columns.str.replace('-', '_')
            hazard_cols = [c for c in hazard_data_details.columns.values.tolist() if c not in ["fname","key"]]
            damage_files = [os.path.join(
                                    asset_damages_results,
                                    f"{country}_{hazard_name}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_direct_damages_parameter_set_{param}.parquet"
                                    ) for param in param_values]
            damage_results = [pd.read_parquet(file) for file in damage_files if os.path.isfile(file) is True]
            # print ("* Done with creating list of all dataframes")
            if damage_results:
                damage_results = pd.concat(damage_results,axis=0,ignore_index=True)
                # print (damage_results)
                for hz in hazard_data_details.itertuples():
                    hz_col = hz.key
                    if hz.key in damage_results.columns.values.tolist():
                        damage = damage_results[[asset_id,"exposure_unit","damage_cost_unit","exposure",hz.key]+uncertainty_columns]
                        damage.rename(columns={hz.key:"damage"},inplace=True)
                        damage["sector"] = asset_info.sector
                        damage["subsector"] = asset_info.asset_gpkg
                        damage["asset_layer"] = asset_info.asset_layer
                        for c in hazard_cols:
                            damage[c] = getattr(hz,c)
                        sum_dict = dict([("exposure","sum"),("damage","sum")])
                        damage_param_sums = damage.groupby(["sector","subsector",
                                                "asset_layer","exposure_unit",
                                                "damage_cost_unit"] + hazard_cols + uncertainty_columns,
                                                dropna=False).agg(sum_dict).reset_index()
                        damage_sensitivity.append(damage_param_sums)
                        del damage_param_sums
                        damage = quantiles(damage,
                                            [asset_id,"sector","subsector",
                                            "asset_layer","exposure_unit",
                                            "damage_cost_unit"] + hazard_cols + ["exposure"],["damage"])
                        damage = damage[damage["damage_amax"]>0]
                        for hk in ["amin","mean","amax"]:     
                            damage[f"exposure_{hk}"] = damage["exposure"]*np.where(damage[f"damage_{hk}"]>0,1,0)
                        damage.drop("exposure",axis=1,inplace=True)
                        asset_damages.append(damage)
                        sum_dict = dict([(f"exposure_{hk}","sum") for hk in ["amin","mean","amax"]]+[(f"damage_{hk}","sum") for hk in ["amin","mean","amax"]])
                        total_damage = damage.groupby(["sector","subsector",
                                            "asset_layer","exposure_unit",
                                            "damage_cost_unit"] + hazard_cols,
                                            dropna=False).agg(sum_dict).reset_index()
                        for hk in ["amin","mean","amax"]:
                            df = damage[damage[f"damage_{hk}"]>0]
                            df = df.groupby(["sector","subsector",
                                            "asset_layer","exposure_unit",
                                            "damage_cost_unit"] + hazard_cols)[asset_id].apply(list).reset_index(name=f"asset_set_{hk}")
                            total_damage = pd.merge(total_damage,df,how="left", on=["sector","subsector",
                                                                                    "asset_layer","exposure_unit",
                                                                                    "damage_cost_unit"] + hazard_cols)
                            del df
                        
                        sector_damages.append(total_damage)

        if asset_damages:
	        asset_damages = pd.concat(asset_damages,axis=0,ignore_index=True)
	        sector_damages = pd.concat(sector_damages,axis=0,ignore_index=True)
	        damage_sensitivity = pd.concat(damage_sensitivity,axis=0,ignore_index=True)
	        asset_damages.to_parquet(os.path.join(summary_results,
	                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages.parquet"),index=False)
	        sector_damages.to_parquet(os.path.join(summary_results,
	                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_sector_damages.parquet"),index=False)
	        damage_sensitivity.to_parquet(os.path.join(summary_results,
	                        f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_damage_sensitivity.parquet"),index=False)

	        # asset_damages.to_csv(os.path.join(summary_results,
	        #                 f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_asset_damages.csv"),index=False)
	        # sector_damages.to_csv(os.path.join(summary_results,
	        #                 f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_sector_damages.csv"),index=False)
	        # damage_sensitivity.to_csv(os.path.join(summary_results,
	        #                 f"{country}_{asset_info.asset_gpkg}_{asset_info.asset_layer}_damage_sensitivity.csv"),index=False)

        print (f"* Done with {asset_info.asset_gpkg} {asset_info.asset_layer}")

if __name__ == "__main__":
    CONFIG = load_config()
    try:
        country =  str(sys.argv[1])
        hazard_names = ast.literal_eval(str(sys.argv[2]))
        direct_damages_folder = str(sys.argv[3])
        summary_results_folder = str(sys.argv[4])
        network_csv = str(sys.argv[5])
        parameter_txt_file = str(sys.argv[6])
    except IndexError:
        print("Got arguments", sys.argv)
        exit()

    main(CONFIG,country,hazard_names,direct_damages_folder,
        summary_results_folder,
        network_csv,
        parameter_txt_file)
