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

def write_to_file(path,filename,key=None):
    df = pd.DataFrame()
    df["path"] = path
    if key is not None:
        df["key"] = key
    df.to_csv(filename,index=False)

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

    countries = ["DMA","GRD","LCA","VCT"]
    data_details = pd.read_csv(os.path.join(
                        processed_data_path,
                        "data_layers",
                        "infrastructure_layers.csv"))
    for country in countries:
        paths = []
        country_files = data_details[data_details["iso_code"] == country]
        for r in country_files.itertuples():
            paths.append(os.path.join("infrastructure",r.sector,r.gpkg))

        country_files = data_details[data_details["iso_code"] == "ALL"]
        for r in country_files.itertuples():       
            data_df = gpd.read_file(
                            os.path.join(
                                processed_data_path,
                                "infrastructure",
                                r.sector,
                                r.gpkg),
                            layer=r.layer
                    )

            # Read the data for each country
            data_df = data_df[data_df["iso_code"] == country]
            data_df.to_file(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                r.sector,
                                f"{country.lower()}_{r.gpkg}"),
                            layer=r.layer,driver="GPKG"
                    )
            paths.append(os.path.join("infrastructure",r.sector,f"{country.lower()}_{r.gpkg}")) 

        # Write the voronoi file path to a csv file
        write_to_file(paths,
                    os.path.join(processed_data_path,"data_layers",f"{country.lower()}_layers.csv"))
        
        """Run the intersections of asset vector layers with the hazard raster grid layers
            This done by calling the script road_raster_intersections.py, which is adapted from:
                https://github.com/nismod/east-africa-transport/blob/main/scripts/exposure/split_networks.py
            The result of this script will give us a geoparquet file with hazard values over geometries of vectors   
        """

        infra_details_csv = os.path.join(processed_data_path,"data_layers",f"{country.lower()}_layers.csv")
        hazards = ["charim_landslide","deltares_storm_surge","fathom_pluvial_fluvial"]
        for hazard in hazards:
            hazard_csv = os.path.join(processed_data_path,"hazards",f"{hazard}_{country.lower()}.csv")

            run_intersections = True  # Set to True is you want to run this process
            # Did a run earlier and it takes ~ 224 minutes (nearly 4 hours) to run for the whole of Africa!
            # And generated a geoparquet with > 24 million rows! 
            if run_intersections is True:
                args = [
                        "python",
                        "vector_raster_intersections.py",
                        f"{infra_details_csv}",
                        f"{hazard_csv}",
                        f"{intersection_results_path}"
                        ]
                print ("* Start the processing of infrastrucutre hazard raster intersections")
                print (args)
                subprocess.run(args)

        print ("* Done with the processing of infrastrucutre hazard raster intersections")



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
