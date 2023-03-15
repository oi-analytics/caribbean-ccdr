""" Assign different weights to road nodes in Africa, including HVT countries
    These weights include:
        Population to whole of Africa 
        Mining areas to whole of Africa
        Sector specific GDP allocations to HVT countries only
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
    df["path"] = [path]
    if key is not None:
        df["key"] = [key]
    df.to_csv(filename,index=False)

def main(config):
    # Set global paths
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    results_data_path = config['paths']['results']

    # Create a path to store voronoi layer outputs
    voronoi_path = os.path.join(processed_data_path,
                            "infrastructure/transport",
                            "roads_voronoi")
    if os.path.exists(voronoi_path) == False:
        os.mkdir(voronoi_path) 

    caribbean_epsg = 32620
    countries = ["DMA","GRD","LCA","VCT"]
    road_nodes = gpd.read_file(
                        os.path.join(
                            processed_data_path,
                            "infrastructure/transport",
                            "roads.gpkg"),
                        layer="nodes"
                    )
    for country in countries:
        # Read the roads nodes for each country
        country_road_nodes = road_nodes[road_nodes["iso_code"] == country]
        country_road_nodes = country_road_nodes.to_crs(epsg=caribbean_epsg)
        # Read the boundary data for each country
        country_boundary = gpd.read_file(
                                os.path.join(
                                    processed_data_path,
                                    "admin_boundaries",
                                    f"gadm41_{country}.gpkg"),
                                layer="ADM_ADM_0")
        country_boundary = country_boundary.to_crs(epsg=caribbean_epsg)
        """Creating Voronoi Polygons for road nodes within an Admin boundary
        """
        min_population = 0
        while min_population == 0:
            roads_voronoi = nodes_voronoi_creations(country_road_nodes,
                                        "node_id",
                                        country_boundary,
                                        epsg=caribbean_epsg)
            roads_voronoi.to_file(
                                os.path.join(
                                        voronoi_path,
                                        f"{country}_roads_voronoi.gpkg"
                                    ),
                                layer="areas",driver="GPKG")

            # Write the voronoi file path to a csv file
            write_to_file(f"infrastructure/transport/roads_voronoi/{country}_roads_voronoi.gpkg",
                        os.path.join(processed_data_path,"data_layers","road_voronoi_layers.csv"))
            # Write the population data layer also to a csv file
            write_to_file(f"socio_economic/population/jrc/2020/population_{country}.tif",
                        os.path.join(processed_data_path,"data_layers","population_layers.csv"),key="pop_2020")

            """Run the intersections of Road Voronoi polygons with the Population raster grid layer
                This done by calling the script road_raster_intersections.py, which is adapted from:
                    https://github.com/nismod/east-africa-transport/blob/main/scripts/exposure/split_networks.py
                The result of this script will give us a geoparquet file with population counts over geometries of Voronoi polygons   
            """

            road_details_csv = os.path.join(processed_data_path,"data_layers","road_voronoi_layers.csv")
            population_details_csv = os.path.join(processed_data_path,"data_layers","population_layers.csv")
            road_pop_intersections_path = voronoi_path

            run_road_pop_intersections = True  # Set to True is you want to run this process
            # Did a run earlier and it takes ~ 224 minutes (nearly 4 hours) to run for the whole of Africa!
            # And generated a geoparquet with > 24 million rows! 
            if run_road_pop_intersections is True:
                args = [
                        "python",
                        "vector_raster_intersections.py",
                        f"{road_details_csv}",
                        f"{population_details_csv}",
                        f"{road_pop_intersections_path}"
                        ]
                print ("* Start the processing of Roads voronoi and population raster intersections")
                print (args)
                subprocess.run(args)

            print ("* Done with the processing of Roads voronoi and population raster intersections")

            """Post-processing the road population intersection result
                The JRC raster layer gives the population-per-pixel (PPP) values
                Assuming each pixel is 100m2, the population denisty in PPP/m2 per pixel is PPP/1.0e4
                The population assigned to the Road Voronoi is (Intersection Area)*PPP/1.0e4
            """
            road_pop_column = "pop_2020" # Name of the Worldpop population column in geoparquet
            road_id_column = "node_id" # Road ID column
            population_grid_area = 1.0e4
            # Read in intersection geoparquet
            road_pop_intersections = gpd.read_parquet(os.path.join(road_pop_intersections_path,
                                        f"{country}_roads_voronoi_splits__population_layers__areas.geoparquet"))
            road_pop_intersections = road_pop_intersections[road_pop_intersections[road_pop_column] > 0]
            road_pop_intersections = road_pop_intersections.to_crs(epsg=caribbean_epsg)
            road_pop_intersections['pop_areas'] = road_pop_intersections.geometry.area
            road_pop_intersections.drop("geometry",axis=1,inplace=True)
            road_pop_intersections[road_pop_column] = road_pop_intersections['pop_areas']*road_pop_intersections[road_pop_column]/population_grid_area
            road_pop_intersections = road_pop_intersections.groupby(road_id_column)[road_pop_column].sum().reset_index()

            roads_voronoi = pd.merge(roads_voronoi,road_pop_intersections,how="left",on=[road_id_column]).fillna(0)
            roads_voronoi.to_file(
                                os.path.join(
                                        voronoi_path,
                                        f"{country}_roads_voronoi.gpkg"
                                    ),
                                layer="areas",driver="GPKG")
            print("* Done with estimating JRC population assinged to each voronoi area in road network")
            # Delete the intersections file to re-run Voronoi population assignment
            os.remove(os.path.join(road_pop_intersections_path,
            		f"{country}_roads_voronoi_splits__population_layers__areas.geoparquet"))
            min_population = min(roads_voronoi[road_pop_column].values)
            filter_nodes = roads_voronoi[roads_voronoi[road_pop_column]>0]["node_id"].values.tolist()
            country_road_nodes = country_road_nodes[country_road_nodes["node_id"].isin(filter_nodes)]



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
