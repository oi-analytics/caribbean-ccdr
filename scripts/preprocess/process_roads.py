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
from tqdm import tqdm
tqdm.pandas()
from utils import *

def components(edges,nodes,node_id_col):
    G = networkx.Graph()
    G.add_nodes_from(
        (getattr(n, node_id_col), {"geometry": n.geometry}) for n in nodes.itertuples()
    )
    G.add_edges_from(
        (e.from_node, e.to_node, {"edge_id": e.edge_id, "geometry": e.geometry})
        for e in edges.itertuples()
    )
    components = networkx.connected_components(G)
    for num, c in enumerate(components):
        print(f"Component {num} has {len(c)} nodes")
        edges.loc[(edges.from_node.isin(c) | edges.to_node.isin(c)), "component"] = num
        nodes.loc[nodes[node_id_col].isin(c), "component"] = num

    return edges, nodes

def get_road_condition_surface(x):
    if not x.surface:
        # if x.highway in ('motorway','trunk','primary'):
        #     return 'paved','asphalt'
        # else:
        #     return 'unpaved','gravel'
        return 'paved','asphalt','model'
    elif x.surface == 'paved':
        return x.surface, 'asphalt', 'OSM'
    elif x.surface == 'unpaved':
        return x.surface, 'gravel','OSM'
    elif x.surface in ('asphalt','concrete'):
        return 'paved',x.surface,'OSM'
    else:
        return 'unpaved',x.surface,'OSM'

def get_road_lanes(x):
    # if there is osm data available, use that and if not assign based on default value
    try:
        float(x.lanes)
        if x.lanes == 0:
            return 1.0,'model'
        else:
            return float(x.lanes),'OSM'
    except (ValueError, TypeError):
        if x.highway in ('motorway','trunk','primary'):
            return 2.0,'model'
        else:
            return 1.0,'model'

def assign_road_speeds(x):
    speed_factor = 1.0
    if x.speed_unit == 'mph':
        speed_factor = 1.61
    if x.highway in ('motorway','trunk','primary'):
        return speed_factor*x["Highway_min"],speed_factor*x["Highway_max"]
    elif x.road_cond == "paved":
        return speed_factor*x["Urban_min"],speed_factor*x["Urban_max"]
    else:
        return speed_factor*x["Rural_min"],speed_factor*x["Rural_max"]

def get_rehab_costs(x, rehab_costs):
    if x.bridge not in ('0','no'): 
        highway = "bridge"
        condition = x.road_cond
    else:
        highway = x.highway
        condition = x.road_cond
    
    cost_min = rehab_costs.cost_min.loc[(rehab_costs.highway==highway)&(rehab_costs.road_cond==condition)].values
    cost_max = rehab_costs.cost_max.loc[(rehab_costs.highway==highway)&(rehab_costs.road_cond==condition)].values
    cost_unit = rehab_costs.cost_unit.loc[(rehab_costs.highway==highway)&(rehab_costs.road_cond==condition)].values
    
    return float(cost_min[0]) , float(cost_max[0]) , str(cost_unit[0])

def add_asset_type(x):
    if x.bridge == "yes":
        return "bridge"
    else:
        return x.road_cond

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    # countries = ['dma','grd','lca','vct']
    # boundaries = pd.concat([gpd.read_file(
    #                 os.path.join(
    #                     incoming_data_path,
    #                     "admin_boundaries",
    #                     f"gadm41_{country}.gpkg"
    #                 ),
    #                 layer="ADM_ADM_0") for country in countries],axis=0,ignore_index=True)
    # boundaries = boundaries.to_crs(epsg=4326)
    # boundaries = boundaries.explode(ignore_index=True)

    # all_edges = []
    # for country in countries:
    #     edges = gpd.read_file(os.path.join(incoming_data_path,
    #                         "hotosm",
    #                         "roads",
    #                         f"hotosm_{country}_roads_gpkg",
    #                         f"hotosm_{country}_roads.gpkg"))
    #     # print (edges)
    #     print("* Done reading file")
    #     # Clean highway names
    #     edges['highway'] = edges['highway'].str.replace('_link','')
    #     # print (list(set(edges.highway.values.tolist())))
    #     edges = edges.to_crs(epsg=4326)
    #     # Calculate and add length of line segments 
    #     geod = Geod(ellps="WGS84")
    #     edges['length_m'] = edges.progress_apply(lambda x:float(geod.geometry_length(x.geometry)),axis=1)
    #     all_edges.append(edges)

    # all_edges = pd.concat(all_edges,axis=0,ignore_index=True)
    # edge_matches = gpd.sjoin(all_edges,
    #                         boundaries, 
    #                         how="left", 
    #                         predicate='intersects'
    #                     ).reset_index()
    # edge_matches = edge_matches.sort_values(by="length_m",ascending=False)
    # edge_matches = edge_matches.drop_duplicates(subset=["osm_id"],keep="first")
    # edge_matches["GID_0"] = edge_matches["GID_0"].astype(str)
    # edge_matches.drop(["index","index_right","COUNTRY","length_m"],axis=1,inplace=True)

    # edge_unmatched = edge_matches[edge_matches["GID_0"] == 'nan']
    # edge_unmatched.drop(["GID_0"],axis=1,inplace=True)
    # edge_unmatched = gpd.sjoin_nearest(edge_unmatched,
    #                             boundaries, 
    #                             how="left").reset_index()
    # edge_unmatched.drop(["index","index_right","COUNTRY"],axis=1,inplace=True)
    # edge_matches = edge_matches[edge_matches["GID_0"] != 'nan']

    # edges = gpd.GeoDataFrame(
    #         pd.concat([edge_matches,edge_unmatched],
    #             axis=0,ignore_index=True),
    #             geometry="geometry",crs="EPSG:4326")
    # print (edges.columns.values.tolist())
    # # Ignore some highway classes
    # edges = edges[~edges['highway'].isin(["steps","path","footway","pedestrian","bridleway","living_street"])]
    # # Create network topology
    # network = create_network_from_nodes_and_edges(
    #     None,
    #     edges,
    #     "road"
    # )

    # network.edges = network.edges.set_crs(epsg=4326)
    # network.nodes = network.nodes.set_crs(epsg=4326)

    # # Add components
    # edges, nodes = components(network.edges,network.nodes,'node_id')
    
    # print (edges.columns.values.tolist())
    # # Calculate and add length of line segments 
    # geod = Geod(ellps="WGS84")
    # edges['length_m'] = edges.progress_apply(lambda x:float(geod.geometry_length(x.geometry)),axis=1)
    
    # # Add road condition and material 
    # edges['surface_material'] = edges.progress_apply(lambda x:get_road_condition_surface(x),axis=1)
    # edges[['road_cond','material','cond_mat_source']] = edges['surface_material'].progress_apply(pd.Series)
    # edges.drop('surface_material',axis=1,inplace=True)

    # # Add number of lanes
    # edges['lanes_infer'] = edges.progress_apply(lambda x:get_road_lanes(x),axis=1)
    # edges[['lanes','lanes_source']] = edges['lanes_infer'].progress_apply(pd.Series)
    # edges.drop('lanes_infer',axis=1,inplace=True)

    # # Assign min and max road speeds
    # road_speeds = pd.read_excel(os.path.join(incoming_data_path,
    #                                 "hotosm",
    #                                 "roads",
    #                                 "road_speeds.xlsx"),sheet_name="speeds")
    # # edges["iso_code"] = str(country).upper()
    # print (edges.columns.values.tolist())
    # edges = pd.merge(edges,road_speeds,how="left",left_on=["gid_0"],right_on=["iso_code"])
    # edges["min_max_speed"] = edges.progress_apply(lambda x:assign_road_speeds(x),axis=1)
    # edges[["min_speed","max_speed"]] = edges["min_max_speed"].progress_apply(pd.Series)
    # edges.drop(["min_max_speed"]+road_speeds.columns.values.tolist(),axis=1,inplace=True)

    # # Assign 'no' to non-bridge edges
    # edges["bridge"] = np.where(edges["bridge"] == 'yes','yes','no')
    # # print (edges)
    # # print (nodes)
    # print("Done adding road attributes")

    # # # Assign road width if needed
    # # width = 3.75 # Default carriageway width in meters for Caribbean, needs to be support by country data
    # # shoulder = 1.5 # Default shoulder width in meters for Caribbean, needs to be support by country data
    # # edges['width_m'] = width*edges['lanes'] + 2.0*shoulder

    # # """Assign cost attributes"""

    # print("Ready to export")
    # # Remove some columns to clean the data
    # edges.drop(["layer","smoothness","surface","source","oneway","width"],axis=1,inplace=True)
    # edges.rename(columns={"gid_0":"iso_code"},inplace=True)
    # edges.to_file(os.path.join(processed_data_path,
    #                         "infrastructure/transport",
    #                         "roads.gpkg"), 
    #                 layer='edges', driver='GPKG')
    # nodes.to_file(os.path.join(processed_data_path,
    #                         "infrastructure/transport",
    #                         "roads.gpkg"), 
    #                 layer='nodes', driver='GPKG')

    # print("Done.")
                       
    # Did not assign is code to roads file. Need to do that
    # edges = gpd.read_file(os.path.join(processed_data_path,
    #                         "infrastructure/transport",
    #                         "roads.gpkg"), 
    #                 layer='edges')
    # nodes = gpd.read_file(os.path.join(processed_data_path,
    #                         "infrastructure/transport",
    #                         "roads.gpkg"), 
    #                 layer='nodes')

    # nodes_iso = list(set(list(
    #                 zip(
    #                     edges["from_node"].values.tolist(),
    #                     edges["iso_code"].values.tolist()
    #                     )
    #                 ) + list(
    #                         zip(
    #                             edges["to_node"].values.tolist(),
    #                             edges["iso_code"].values.tolist()
    #                             )
    #                     )
    #                 )
    #             )
    # nodes_iso = pd.DataFrame(nodes_iso,columns=["node_id","iso_code"])
    # nodes = pd.merge(nodes,nodes_iso,how="left",on=["node_id"])
    # gpd.GeoDataFrame(nodes,
    #             geometry="geometry",
    #             crs="EPSG:4326").to_file(os.path.join(processed_data_path,
    #                         "infrastructure/transport",
    #                         "roads.gpkg"), 
    #                 layer='nodes', driver='GPKG')

    countries = ['dma','grd','lca','vct']
    for country in countries:
        edges = gpd.read_file(os.path.join(processed_data_path,
                                "infrastructure/transport",
                                f"{country}_roads.gpkg"), 
                        layer='edges')
        edges["asset_type"] = edges.progress_apply(lambda x:add_asset_type(x),axis=1)
        edges.to_file(os.path.join(processed_data_path,
                                "infrastructure/transport",
                                f"{country}_roads.gpkg"), 
                        layer='edges',driver="GPKG")  

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
