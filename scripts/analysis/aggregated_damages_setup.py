"""This script allows us to select and parallelise the Damage and Loss estimations on a server with multiple core processors
"""
import os
import sys
import ujson
import itertools
import geopandas as gpd
import shutil
from analysis_utils import *
import subprocess 

def main(config):
    processed_data_path = config['paths']['data']
    results_path = config['paths']['results']

    adaptation_options = ["no_adaptation","with_adaptation"]
    countries = ["dma","grd","lca","vct"]
    countries = ["grd"]
    hazards = ["charim_landslide","deltares_storm_surge","fathom_pluvial_fluvial","storm_cyclones"]
    # hazard_columns = ["hazard","isoa3","epoch","rcp","rp","precipitation_factor"]
    # edge_id sector  subsector   asset_layer exposure_unit   damage_cost_unit    hazard  isoa3   epoch   rcp rp
    hazard_columns = ["sector","subsector","damage_cost_unit","isoa3","epoch","rcp","rp"]
    development_scenarios = ["bau","sdg"]
    for adapt_option in adaptation_options:
        results_folder = os.path.join(results_path,adapt_option)
        if os.path.exists(results_folder) == False:
            os.mkdir(results_folder)
        
        summary_folder = f"{adapt_option}/direct_damages"
        for country in countries:
            if country == "grd":
                network_csv = os.path.join(processed_data_path,
                            "data_layers",
                            "network_layers_hazard_intersections_details_grd.csv")
            else:
                network_csv = os.path.join(processed_data_path,
                            "data_layers",
                            "network_layers_hazard_intersections_details.csv")

            for dev_sc in development_scenarios:
                args = [
                        "python",
                        "hazard_aggregated_damages.py",
                        f"{country}",
                        f"{hazard_columns}",
                        f"{summary_folder}",
                        f"{network_csv}",
                        f"{dev_sc}"
                        ]
                print ("* Start the processing of summarising damage results")
                print (args)
                subprocess.run(args)
                if adapt_option == "with_adaptation":
                    args = [
                        "python",
                        "aggregated_benefits.py",
                        f"{country}",
                        f"{dev_sc}"
                        ]
                    print ("* Start the processing of combining damage results")
                    print (args)
                    subprocess.run(args)

                    args = [
                        "python",
                        "aggregated_results_combined.py",
                        f"{country}",
                        f"{dev_sc}"
                        ]
                    print ("* Start the processing of summarising damage results")
                    print (args)
                    subprocess.run(args)






                                
if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)