""" Assign populations to energy nodes in East Caribbean
    This is done by creating Voronoi polygons around energy and intersecting with population rasters
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

def write_to_file(path,path_column,filename,key=None,layer=None):
    df = pd.DataFrame()
    df[path_column] = [path]
    if key is not None:
        df["key"] = [key]
    if layer is not None:
        df["layer"] = [layer]
    df.to_csv(filename,index=False)

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
    voronoi_path = os.path.join(processed_data_path,
                            "infrastructure/energy",
                            "energy_voronoi")
    if os.path.exists(voronoi_path) == False:
        os.mkdir(voronoi_path)

    energy_flow_path = os.path.join(results_data_path,
                            "energy_flow_paths")
    if os.path.exists(energy_flow_path) == False:
        os.mkdir(energy_flow_path) 

    caribbean_epsg = 32620
    countries = ["dma","lca","vct"]
    countries_access = [1.0,0.9955,1.0]
    countries = ["vct"]
    countries_access = [1.0]
    for cdx,(country,access) in enumerate(zip(countries,countries_access)):
        # Read the energy nodes for each country
        read_gpkg = os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "energy",
                            f"{country.lower()}_energy.gpkg")
        if os.path.isfile(read_gpkg):
            layers = [l for l in fiona.listlayers(read_gpkg) if l in ["nodes","areas"]]
            sinks_df = []
            sources_df = []
            for layer in layers:
                gdf = gpd.read_file(read_gpkg,layer=layer)
                gdf = gdf.to_crs(epsg=caribbean_epsg)
                if layer == "areas":
                    gdf["geometry"] = gdf.geometry.centroid
                sinks = gdf[gdf["function_type"] == "sink"]
                sources = gdf[gdf["function_type"] == "source"]
                if len(sinks.index) > 0:
                    sinks_df.append(sinks[["node_id","geometry"]])

                if len(sources.index) > 0:   
                    sources_df.append(sources[["node_id","asset_type","capacity_mw"]])
            if sinks_df and sources_df:
                sinks_df = gpd.GeoDataFrame(
                                pd.concat(sinks_df,axis=0,ignore_index=True),
                                geometry="geometry",
                                crs=f"EPSG:{caribbean_epsg}")

                sources_df = pd.concat(sources_df,axis=0,ignore_index=True)
                # Read the boundary data for each country
                country_boundary = gpd.read_file(
                                        os.path.join(
                                            processed_data_path,
                                            "admin_boundaries",
                                            f"gadm41_{country.upper()}.gpkg"),
                                        layer="ADM_ADM_0")
                country_boundary = country_boundary.to_crs(epsg=caribbean_epsg)
                """Creating Voronoi Polygons for energy nodes within an Admin boundary
                """
                energy_voronoi = nodes_voronoi_creations(sinks_df,
                                            "node_id",
                                            country_boundary,
                                            epsg=caribbean_epsg)
                gpd.GeoDataFrame(energy_voronoi,geometry="geometry",crs=f"EPSG:{caribbean_epsg}").to_file(
                                    os.path.join(
                                            voronoi_path,
                                            f"{country}_energy_voronoi.gpkg"
                                        ),
                                    layer="areas",driver="GPKG")

                # Write the voronoi file path to a csv file
                write_to_file(f"infrastructure/energy/energy_voronoi/{country}_energy_voronoi.gpkg","path",
                            os.path.join(processed_data_path,"data_layers","energy_voronoi_layers.csv"),layer="areas")
                # Write the population data layer also to a csv file
                write_to_file(f"socio_economic/population/jrc/2020/population_{country}.tif","path",
                            os.path.join(processed_data_path,"data_layers","population_layers.csv"),key="pop_2020")

                """Run the intersections of Energy Voronoi polygons with the Population raster grid layer
                    This done by calling the script vector_raster_intersections.py, which is adapted from:
                        https://github.com/nismod/east-africa-transport/blob/main/scripts/exposure/split_networks.py
                    The result of this script will give us a geoparquet file with population counts over geometries of Voronoi polygons   
                """

                energy_details_csv = os.path.join(processed_data_path,"data_layers","energy_voronoi_layers.csv")
                population_details_csv = os.path.join(processed_data_path,"data_layers","population_layers.csv")
                features_df = create_feature_csv(energy_details_csv,population_details_csv,voronoi_path)
                features_csv = os.path.join(processed_data_path,"data_layers",f"{country}_energy_voronoi_layers.csv")
                features_df.to_csv(features_csv,index=False)
                
                run_energy_pop_intersections = True  # Set to True is you want to run this process
                # Did a run earlier and it takes ~ 224 minutes (nearly 4 hours) to run for the whole of Africa!
                # And generated a geoparquet with > 24 million rows! 
                if run_energy_pop_intersections is True:
                    args = [
                            "env", 
                            "SNAIL_PROGRESS=1",
                            "snail",
                            "-vv",
                            "process",
                            "--features",
                            f"{features_csv}",
                            "--rasters",
                            f"{population_details_csv}",
                            "--directory",
                            f"{processed_data_path}"
                            ]
                    print ("* Start the processing of energy voronoi and population raster intersections")
                    print (args)
                    subprocess.run(args)

                print ("* Done with the processing of energy voronoi and population raster intersections")

                """Post-processing the energy population intersection result
                    The JRC raster layer gives the population-per-pixel (PPP) values
                    Assuming each pixel is 100m2, the population denisty in PPP/m2 per pixel is PPP/1.0e4
                    The population assigned to the energy Voronoi is (Intersection Area)*PPP/1.0e4
                """
                energy_pop_column = "pop_2020" # Name of the Worldpop population column in geoparquet
                energy_id_column = "node_id" # energy ID column
                population_grid_area = 1.0e4
                # Read in intersection geoparquet
                energy_pop_intersections = gpd.read_parquet(os.path.join(voronoi_path,
                                            f"{country}_energy_voronoi_splits__population_layers__areas.geoparquet"))
                energy_pop_intersections = energy_pop_intersections[energy_pop_intersections[energy_pop_column] > 0]
                energy_pop_intersections = energy_pop_intersections.to_crs(epsg=caribbean_epsg)
                energy_pop_intersections['pop_areas'] = energy_pop_intersections.geometry.area
                energy_pop_intersections.drop("geometry",axis=1,inplace=True)
                energy_pop_intersections[energy_pop_column] = energy_pop_intersections['pop_areas']*energy_pop_intersections[energy_pop_column]/population_grid_area
                energy_pop_intersections = energy_pop_intersections.groupby(energy_id_column)[energy_pop_column].sum().reset_index()

                energy_voronoi = pd.merge(energy_voronoi,energy_pop_intersections,how="left",on=[energy_id_column]).fillna(0)
                energy_voronoi[energy_pop_column] = access*energy_voronoi[energy_pop_column]
                # gpd.GeoDataFrame(energy_voronoi,geometry="geometry",crs=f"EPSG:{caribbean_epsg}").to_file(
                #                     os.path.join(
                #                             voronoi_path,
                #                             f"{country}_energy_voronoi.gpkg"
                #                         ),
                #                     layer="areas",driver="GPKG")
                print("* Done with estimating JRC population assinged to each voronoi area in energy network")
                # Delete the intersections file to re-run Voronoi population assignment
                os.remove(os.path.join(voronoi_path,
                        f"{country}_energy_voronoi_splits__population_layers__areas.geoparquet"))


                edges = gpd.read_file(read_gpkg,layer="edges")[['from_node','to_node','edge_id']]
                sources_df.rename(columns={"node_id":"origin_id"},inplace=True)
                energy_voronoi.rename(columns={"node_id":"destination_id"},inplace=True)
                od_matrix = sources_df[["origin_id","asset_type","capacity_mw"]].merge(energy_voronoi[["destination_id",energy_pop_column]], how='cross')
                
                # od_matrix = pd.DataFrame([(x,y) for x in sources_df.iterrows() for y in energy_voronoi.iterrows()],columns=['origin_id','destination_id'])
                
                print ('start flow mapping')
                # get_all_od_paths(graph,sources,sinks,cutoff=100,directed=True,csv_path=os.path.join(BASE_PATH,'path_mapping','electricity_paths.csv'))
                all_paths = network_ods_assembly(od_matrix,edges,
                                None,["capacity_mw",energy_pop_column],directed=False,
                                file_output_path=os.path.join(energy_flow_path,f"{country}_energy_paths.parquet"))
                # all_paths.to_csv(os.path.join(energy_flow_path,f"{country}_energy_paths.csv"),index=False)
                print ('Total number of paths',len(all_paths.index))



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
