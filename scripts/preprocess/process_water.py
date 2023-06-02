#!/usr/bin/env python
# coding: utf-8
"""Process road data from OSM extracts and create road network topology 
    WILL MODIFY LATER
"""
import os
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
from pyproj import Geod
import networkx
import igraph
import snkit
import subprocess
from shapely.ops import linemerge
from shapely.geometry import Point,LineString,MultiLineString
from tqdm import tqdm
tqdm.pandas()
from utils import *
from spatial_utils import nodes_voronoi_creations

def write_to_file(path,path_column,filename,key=None):
    df = pd.DataFrame()
    df[path_column] = [path]
    if key is not None:
        df["key"] = [key]
    df.to_csv(filename,index=False)

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    caribbean_crs = 32620

    for asset_type in ["wtp","wwtp"]: 
        lca_df = gpd.read_file(os.path.join(incoming_data_path,
                                "water_data",
                                f"lca_{asset_type}.gpkg"),layer="nodes")

        lca_df = lca_df.to_crs(epsg=caribbean_crs)
        lca_df.rename(columns={"Capacity":"capacity_m3d",
                            "Access Value":"capacity_m3d",
                            "% Population":"capacity_percentage"},
                        inplace=True)
        lca_df["asset_type"] = asset_type
        lca_df["capacity_m3d"] = lca_df["capacity_m3d"]/365.0
        
        use_voronoi_method = False
        if asset_type == "wtp":
            if use_voronoi_method is False:
                total_pop_served = 0.9689*179667 # assumed coverage multiplied by 2018 population 
                total_cap  = lca_df[lca_df["Type"] != "Desal plant"]["capacity_m3d"].sum()
                print (total_cap)
                lca_df["estimated_population"] = total_pop_served*lca_df["capacity_m3d"]/total_cap
                lca_df.loc[lca_df["Type"] == "Desal plant","estimated_population"] = 1000
            else:
                # Read the boundary data for each country
                # Create a path to store voronoi layer outputs
                voronoi_path = os.path.join(processed_data_path,
                                        "infrastructure/water",
                                        "water_voronoi")
                if os.path.exists(voronoi_path) == False:
                    os.mkdir(voronoi_path)
                country_boundary = gpd.read_file(
                                        os.path.join(
                                            processed_data_path,
                                            "admin_boundaries",
                                            f"gadm41_LCA.gpkg"),
                                        layer="ADM_ADM_0")
                country_boundary = country_boundary.to_crs(epsg=caribbean_crs)
                """Creating Voronoi Polygons for water nodes within an Admin boundary
                """
                water_voronoi = nodes_voronoi_creations(lca_df,
                                            "node_id",
                                            country_boundary,
                                            epsg=caribbean_crs)
                water_voronoi.to_file(
                                    os.path.join(
                                            voronoi_path,
                                            "lca_water_voronoi.gpkg"
                                        ),
                                    layer="areas",driver="GPKG")

                # Write the voronoi file path to a csv file
                write_to_file(f"infrastructure/water/water_voronoi/lca_water_voronoi.gpkg","path",
                            os.path.join(processed_data_path,"data_layers","water_voronoi_layers.csv"))
                # Write the population data layer also to a csv file
                write_to_file(f"socio_economic/population/jrc/2020/population_lca.tif","fname",
                            os.path.join(processed_data_path,"data_layers","population_layers.csv"),key="pop_2020")

                """Run the intersections of water Voronoi polygons with the Population raster grid layer
                    This done by calling the script vector_raster_intersections.py, which is adapted from:
                        https://github.com/nismod/east-africa-transport/blob/main/scripts/exposure/split_networks.py
                    The result of this script will give us a geoparquet file with population counts over geometries of Voronoi polygons   
                """

                water_details_csv = os.path.join(processed_data_path,"data_layers","water_voronoi_layers.csv")
                population_details_csv = os.path.join(processed_data_path,"data_layers","population_layers.csv")
                
                run_water_pop_intersections = True  # Set to True is you want to run this process
                # Did a run earlier and it takes ~ 224 minutes (nearly 4 hours) to run for the whole of Africa!
                # And generated a geoparquet with > 24 million rows! 
                if run_water_pop_intersections is True:
                    args = [
                            "python",
                            "vector_raster_intersections.py",
                            f"{water_details_csv}",
                            f"{population_details_csv}",
                            f"{voronoi_path}"
                            ]
                    print ("* Start the processing of water voronoi and population raster intersections")
                    print (args)
                    subprocess.run(args)

                print ("* Done with the processing of water voronoi and population raster intersections")

                """Post-processing the water population intersection result
                    The JRC raster layer gives the population-per-pixel (PPP) values
                    Assuming each pixel is 100m2, the population denisty in PPP/m2 per pixel is PPP/1.0e4
                    The population assigned to the water Voronoi is (Intersection Area)*PPP/1.0e4
                """
                water_pop_column = "pop_2020" # Name of the Worldpop population column in geoparquet
                water_id_column = "node_id" # water ID column
                population_grid_area = 1.0e4
                # Read in intersection geoparquet
                water_pop_intersections = gpd.read_parquet(os.path.join(voronoi_path,
                                            f"lca_water_voronoi_splits__population_layers__areas.geoparquet"))
                water_pop_intersections = water_pop_intersections[water_pop_intersections[water_pop_column] > 0]
                water_pop_intersections = water_pop_intersections.to_crs(epsg=caribbean_crs)
                water_pop_intersections['pop_areas'] = water_pop_intersections.geometry.area
                water_pop_intersections.drop("geometry",axis=1,inplace=True)
                water_pop_intersections[water_pop_column] = water_pop_intersections['pop_areas']*water_pop_intersections[water_pop_column]/population_grid_area
                water_pop_intersections = water_pop_intersections.groupby(water_id_column)[water_pop_column].sum().reset_index()

                water_voronoi = pd.merge(water_voronoi,water_pop_intersections,how="left",on=[water_id_column]).fillna(0)
                water_voronoi.to_file(
                                    os.path.join(
                                            voronoi_path,
                                            f"lca_water_voronoi.gpkg"
                                        ),
                                    layer="areas",driver="GPKG")
                print("* Done with estimating JRC population assinged to each voronoi area in water network")
                # Delete the intersections file to re-run Voronoi population assignment
                os.remove(os.path.join(voronoi_path,
                        f"lca_water_voronoi_splits__population_layers__areas.geoparquet"))

        lca_df.to_file(os.path.join(processed_data_path,
                                        "infrastructure/water",
                                        f"lca_{asset_type}.gpkg"),
                                        layer="nodes",
                                        driver="GPKG")

    grd_df = gpd.read_file(os.path.join(incoming_data_path,
                                "water_data",
                                "grd_water.gpkg"))
    grd_df = grd_df.to_crs(epsg=caribbean_crs)
    grd_df["node_id"] = grd_df.index.values.tolist()
    grd_df["node_id"] = grd_df.progress_apply(lambda x:f"GRD_wtp_{x.node_id}",axis=1)
    grd_df["asset_type"] = "wtp"
    grd_df["capacity_m3d"] = grd_df["capacity_mgd"]*3785.0

    grd_tot_cap = pd.read_excel(os.path.join(incoming_data_path,
                                    "water_data",
                                    "grd_wtw_capacities.xlsx"),sheet_name="wtw_capacity")["capacity_mgd"].sum()
    grd_tot_pop = 0.98*124610 # percent access X 2021 population statistics
    grd_df["estimated_population"] = grd_tot_pop*grd_df["capacity_mgd"]/grd_tot_cap
    grd_df.to_file(os.path.join(processed_data_path,
                                        "infrastructure/water",
                                        "grd_wtp.gpkg"),
                                        layer="nodes",
                                        driver="GPKG")


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
