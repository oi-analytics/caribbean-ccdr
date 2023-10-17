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

    damage_results_folder = "intermediate_damages"
    summary_folder = "direct_damages"
    service_loss_folder = "damage_service_losses"

    # network_csv = os.path.join(processed_data_path,
    #                         "data_layers",
    #                         "network_layers_hazard_intersections_details.csv")
    service_csv = os.path.join(processed_data_path,
                            "data_layers",
                            "service_growth_rates.csv")
    hazard_damage_parameters_csv = os.path.join(processed_data_path,
                            "damage_curves",
                            "hazard_damage_parameters.csv")
    damage_curves_csv = os.path.join(processed_data_path,
                            "damage_curves",
                            "asset_damage_curve_mapping.csv")
    adaptation_options = ["no_adaptation","with_adaptation"]
    # adaptation_options = ["no_adaptation"]
    # adaptation_options = ["with_adaptation"]
    countries = ["dma","grd","lca","vct"]
    countries = ["vct"]
    # hazards = ["charim_landslide","deltares_storm_surge","fathom_pluvial_fluvial","chaz_cyclones"]
    hazards = ["charim_landslide","deltares_storm_surge","fathom_pluvial_fluvial","storm_cyclones"]
    # hazard_columns = ["hazard","isoa3","epoch","rcp","rp","precipitation_factor"]
    hazard_columns = ["hazard","isoa3","epoch","rcp","rp"]
    development_scenarios = ["bau","sdg"]
    # development_scenarios = ["sdg"]
    parameter_combinations_file = "parameter_combinations.txt"
    generate_new_parameters = True
    if generate_new_parameters is True:
	    # Set up problem for sensitivity analysis
	    problem = {
	              'num_vars': 2,
	              'names': ['cost_uncertainty_parameter','damage_uncertainty_parameter'],
	              'bounds': [[0.0,0.5,1.0],[0.0,0.5,1.0]]
	              }
	    
	    # And create parameter values
	    param_values = list(itertools.product(*problem['bounds']))
	    with open(parameter_combinations_file,"w+") as f:
	        for p in range(len(param_values)):  
	            f.write(f"{p},{param_values[p][0]},{param_values[p][1]}\n")
	    
	    f.close()
    for adapt_option in adaptation_options:
        results_folder = os.path.join(results_path,adapt_option)
        if os.path.exists(results_folder) == False:
            os.mkdir(results_folder)
        
        damage_results_folder = f"{adapt_option}/intermediate_damages"
        summary_folder = f"{adapt_option}/direct_damages"
        service_loss_folder = f"{adapt_option}/damage_service_losses"
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
                for hazard_name in hazards:
                    hazard_csv = os.path.join(processed_data_path,
                                            "hazards",
                                            f"{hazard_name}_{country}.csv")
                    with open("damage_results.txt","w+") as f:
                        with open(parameter_combinations_file,"r") as r:
                            for p in r:
                                pv = p.split(",")
                                f.write(f"{country},{hazard_name},{damage_results_folder},{network_csv},{hazard_csv},{damage_curves_csv},{hazard_damage_parameters_csv},{adapt_option},{dev_sc},{pv[0]},{pv[1]},{pv[2]}\n")
                    
                    f.close()

                    num_blocks = len(param_values)
                    """Next we call the failure analysis script and loop through the failure scenarios
                    """
                    args = ["parallel",
                            "-j", str(num_blocks),
                            "--colsep", ",",
                            "-a",
                            "damage_results.txt",
                            "python",
                            "damage_calculations.py",
                            "{}"
                            ]
                    print ("* Start the processing of damage calculations")
                    print (args)
                    subprocess.run(args)

                """Next we call the summary scripts
                """
                args = [
                        "python",
                        "damages_summarise.py",
                        f"{country}",
                        f"{hazards}",
                        f"{hazard_columns}",
                        f"{damage_results_folder}",
                        f"{summary_folder}",
                        f"{network_csv}",
                        f"{parameter_combinations_file}",
                        f"{dev_sc}"
                        ]
                print ("* Start the processing of summarising damage results")
                print (args)
                subprocess.run(args)

                """Next we remove the damage results folder because we do not want all those files
                """
                shutil.rmtree(os.path.join(results_path,f"{damage_results_folder}_{dev_sc}"))

                """Next we call the losses scripts
                """
                if adapt_option == "no_adaptation":
                    args = [
                            "python",
                            "asset_service_disruption.py",
                            f"{country}",
                            f"{hazard_columns}",
                            f"{summary_folder}",
                            f"{service_loss_folder}",
                            f"{network_csv}",
                            f"{service_csv}",
                            f"{dev_sc}"
                            ]
                    print ("* Start the processing of summarising damage results")
                    print (args)
                    subprocess.run(args)
                    hazard_indexes = hazard_columns 
                else:
                    args = [
                            "python",
                            "adaptation_prioritisation.py",
                            f"{country}",
                            f"{hazard_columns}",
                            f"{summary_folder}",
                            f"{service_loss_folder}",
                            f"{network_csv}",
                            f"{dev_sc}"
                            ]
                    print ("* Start the processing of summarising damage results")
                    print (args)
                    subprocess.run(args)
                    hazard_indexes = hazard_columns + ["service_resilience_target_percentage"]

                args = [
                        "python",
                        "service_disruptions.py",
                        f"{country}",
                        f"{hazard_indexes}",
                        f"{summary_folder}",
                        f"{service_loss_folder}",
                        f"{network_csv}",
                        f"{service_csv}",
                        f"{dev_sc}",
                        f"{adapt_option}"
                        ]
                print ("* Start the processing of summarising damage results")
                print (args)
                subprocess.run(args)

                if adapt_option == "with_adaptation":
                    args = [
                        "python",
                        "benefits_estimation.py",
                        f"{country}",
                        f"{network_csv}",
                        f"{dev_sc}"
                        ]
                    print ("* Start the processing of summarising damage results")
                    print (args)
                    subprocess.run(args)

                    args = [
                        "python",
                        "combined_results.py",
                        f"{country}",
                        f"{network_csv}",
                        f"{dev_sc}"
                        ]
                    print ("* Start the processing of summarising damage results")
                    print (args)
                    subprocess.run(args)




                                
if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)