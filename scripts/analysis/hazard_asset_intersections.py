""" Intersect assets with hazard maps
"""

import os
import fiona
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point,LineString,Polygon
from shapely.ops import nearest_points
from scipy.spatial import Voronoi, cKDTree
import subprocess
from analysis_utils import *
from tqdm import tqdm
tqdm.pandas()

def create_feature_csv(networks_csv, hazards_csv,output_path):
    # read transforms, record with hazards
    output_df = []
    hazards = pd.read_csv(hazards_csv)
    hazards.rename(columns={"fname":"path"},inplace=True)
    hazards.to_csv(hazards_csv,index=False)
    if len(hazards.index) > 0:
        hazard_slug = os.path.basename(hazards_csv).replace(".csv", "")
        # hazard_transforms, transforms = read_transforms(hazards, data_path)
        # hazard_transforms.to_csv(hazards_csv.replace(".csv", "__with_transforms.csv"), index=False)

        # read networks
        networks = pd.read_csv(networks_csv)

        for n in networks.itertuples():
            network_path = n.path
            layer_name = n.layer
            # fname = os.path.join(data_path, network_path)
            out_fname = os.path.join(
                output_path,
                os.path.basename(network_path).replace(".gpkg", f"_splits__{hazard_slug}.gpkg")
            )
            fname_out = out_fname.replace(".gpkg",f"__{layer_name}.geoparquet")
            output_df.append((network_path,layer_name,fname_out))

    return pd.DataFrame(output_df,columns=["path","layer","output_path"])
            

def main(config):
    # Set global paths
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    results_data_path = config['paths']['results']

    # Create a path to store voronoi layer outputs
    intersection_results_path = os.path.join(results_data_path,
                            "hazard_asset_intersections")
    if os.path.exists(intersection_results_path) == False:
        os.mkdir(intersection_results_path) 

    # countries = ["DMA","GRD","LCA","VCT"]
    # data_details = pd.read_csv(os.path.join(
    #                     processed_data_path,
    #                     "data_layers",
    #                     "infrastructure_layers.csv"))
    countries = ["dma","grd","lca","vct"]
    countries = ["grd"]
    sectors = {
                "transport":["roads","ports","airports"],
                "energy":["energy"],
                "water":["wtp","wwtp"],
                "social":["education","health"]
            }
    # sectors = {
    #             # "energy":["energy"],
    #             "water":["wtp","wwtp"],
    #             "social":["education","health"],
    #             "transport":["airports","ports"]
    #         }
    # sectors = {
    #             # "energy":["energy"],
    #             "transport":["ports"]
    #         }
    for country in countries:
        paths = []
        for sector,subsectors in sectors.items():
            for subsector in subsectors:
                read_gpkg = os.path.join(
                                    processed_data_path,
                                    "infrastructure",
                                    sector,
                                    f"{country}_{subsector}.gpkg")
                if os.path.isfile(read_gpkg):
                    layers = fiona.listlayers(read_gpkg)
                    for layer in layers:
                        paths.append((os.path.join("infrastructure",sector,f"{country}_{subsector}.gpkg"),layer))

        paths = pd.DataFrame(paths,columns = ["path","layer"])
        paths.to_csv(os.path.join(processed_data_path,"data_layers",f"{country}_layers.csv"),index=False) 
        
        # """Run the intersections of asset vector layers with the hazard raster grid layers
        #     This done by calling the script vector_raster_intersections.py, which is adapted from:
        #         https://github.com/nismod/east-africa-transport/blob/main/scripts/exposure/split_networks.py
        #     The result of this script will give us a geoparquet file with hazard values over geometries of vectors   
        # """

        infra_details_csv = os.path.join(processed_data_path,"data_layers",f"{country}_layers.csv")
        # hazards = ["charim_landslide","deltares_storm_surge","fathom_pluvial_fluvial","chaz_cyclones"]
        hazards = ["storm_cyclones","deltares_storm_surge","fathom_pluvial_fluvial","charim_landslide"]
        # hazards = ["charim_landslide"]
        for hazard in hazards:
            hazard_csv = os.path.join(processed_data_path,"hazards",f"{hazard}_{country}.csv")
            features_df = create_feature_csv(infra_details_csv,hazard_csv,intersection_results_path)
            features_csv = os.path.join(processed_data_path,"data_layers",f"{country}_{hazard}_layers.csv")
            features_df.to_csv(features_csv,index=False)

            run_intersections = True  # Set to True is you want to run this process
            if run_intersections is True:
                args = [
                        "env", 
                        "SNAIL_PROGRESS=1",
                        "snail",
                        "-vv",
                        "process",
                        "--features",
                        f"{features_csv}",
                        "--rasters",
                        f"{hazard_csv}",
                        "--directory",
                        f"{processed_data_path}"
                        ]
                print ("* Start the processing of infrastrucutre hazard raster intersections")
                print (args)
                subprocess.run(args)

        print ("* Done with the processing of infrastrucutre hazard raster intersections")



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
