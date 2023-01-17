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

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    countries = ['dma','grd','lca','vct']
    for country in countries:
        edges = gpd.read_file(os.path.join(incoming_data_path,
                            "hotosm",
                            "roads",
                            f"hotosm_{country}_roads_gpkg",
                            f"hotosm_{country}_roads.gpkg"))
        # print (edges)
        print("* Done reading file")
        
        # Ignore some highway classes
        edges = edges[~edges['highway'].isin(["steps","path","footway","pedestrian","bridleway","living_street"])]
        # Clean highway names
        edges['highway'] = edges['highway'].str.replace('_link','')
        # print (list(set(edges.highway.values.tolist())))
    
        # Create network topology
        network = create_network_from_nodes_and_edges(
            None,
            edges,
            f"{country.upper()}_road"
        )

        network.edges = network.edges.set_crs(epsg=4326)
        network.nodes = network.nodes.set_crs(epsg=4326)

        # Add components
        edges, nodes = components(network.edges,network.nodes,'node_id')
        
        # Calculate and add length of line segments 
        geod = Geod(ellps="WGS84")
        edges['length_m'] = edges.progress_apply(lambda x:float(geod.geometry_length(x.geometry)),axis=1)
        
        # Add road condition and material 
        edges['surface_material'] = edges.progress_apply(lambda x:get_road_condition_surface(x),axis=1)
        edges[['road_cond','material','cond_mat_source']] = edges['surface_material'].progress_apply(pd.Series)
        edges.drop('surface_material',axis=1,inplace=True)

        # Add number of lanes
        edges['lanes_infer'] = edges.progress_apply(lambda x:get_road_lanes(x),axis=1)
        edges[['lanes','lanes_source']] = edges['lanes_infer'].progress_apply(pd.Series)
        edges.drop('lanes_infer',axis=1,inplace=True)

        # Assign min and max road speeds
        road_speeds = pd.read_excel(os.path.join(incoming_data_path,
                                        "hotosm",
                                        "roads",
                                        "road_speeds.xlsx"),sheet_name="speeds")
        edges["iso_code"] = str(country).upper()
        edges = pd.merge(edges,road_speeds,how="left",on=["iso_code"])
        edges["min_max_speed"] = edges.progress_apply(lambda x:assign_road_speeds(x),axis=1)
        edges[["min_speed","max_speed"]] = edges["min_max_speed"].progress_apply(pd.Series)
        edges.drop(["min_max_speed"]+road_speeds.columns.values.tolist(),axis=1,inplace=True)

        # Assign 'no' to non-bridge edges
        edges["bridge"] = np.where(edges["bridge"] == 'yes','yes','no')
        # print (edges)
        # print (nodes)
        print("Done adding road attributes")

        # # Assign road width if needed
        # width = 3.75 # Default carriageway width in meters for Caribbean, needs to be support by country data
        # shoulder = 1.5 # Default shoulder width in meters for Caribbean, needs to be support by country data
        # edges['width_m'] = width*edges['lanes'] + 2.0*shoulder

        # """Assign cost attributes"""

        print("Ready to export")
        # Remove some columns to clean the data
        edges.drop(["layer","smoothness","surface","source","oneway","width"],axis=1,inplace=True)
        edges.to_file(os.path.join(processed_data_path,
                                "infrastructure/transport",
                                f"{country}_roads.gpkg"), 
                        layer='edges', driver='GPKG')
        nodes.to_file(os.path.join(processed_data_path,
                                "infrastructure/transport",
                                f"{country}_roads.gpkg"), 
                        layer='nodes', driver='GPKG')

        print("Done.")
                       


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
