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
from shapely.geometry import Point,LineString
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

def network_degree(edges,nodes,node_id_col):
    G = networkx.Graph()
    G.add_nodes_from(
        (getattr(n, node_id_col), {"geometry": n.geometry}) for n in nodes.itertuples()
    )
    G.add_edges_from(
        (e.from_node, e.to_node, {"edge_id": e.edge_id, "geometry": e.geometry})
        for e in edges.itertuples()
    )
    degrees = pd.DataFrame([(node,val) for (node, val) in G.degree()],columns=[node_id_col,"degree"])
    nodes = pd.merge(nodes,degrees,how="left",on=[node_id_col])
    return nodes

def add_new_edges(new_edges,existing_edges,existing_nodes,max_edge_id,epsg=32360):
    new_edges = pd.merge(new_edges,existing_nodes[["node_id","geometry"]],how="left",on=["node_id"])
    new_edges.rename(columns={"node_id":"to_node","geometry":"to_geometry"},inplace=True)
    new_edges["geometry"] = new_edges.progress_apply(lambda x:LineString([x.from_geometry,x.to_geometry]),axis=1)
    new_edges = gpd.GeoDataFrame(new_edges,geometry="geometry",crs=f"EPSG:{epsg}")
    new_edges["edge_id"] = list(max_edge_id + 1 + new_edges.index.values)
    new_edges["edge_id"] = new_edges.progress_apply(lambda x: f"elece_{x.edge_id}",axis=1)
    new_edges["length_m"] = new_edges.geometry.length
    new_edges.drop(["from_geometry","to_geometry"],axis=1,inplace=True)
    new_edges["feedernm"] = "dummy"
    new_edges["shape_leng"] = new_edges["length_m"]
    new_edges["objectid"] = list(max(existing_edges.objectid.values.tolist()) + 1 + new_edges.index.values)
    # new_edges = gpd.GeoDataFrame(new_edges,geometry="geometry",crs=f"EPSG:{epsg}")
    return new_edges

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    plants_substations = gpd.read_file(os.path.join(incoming_data_path,"st_lucia_energy","stlucelectricity.shp"))
    plants_substations.rename(columns={"type":"asset_type"},inplace=True)
    plants_substations = plants_substations.to_crs(epsg=32620)

    poles = gpd.read_file(os.path.join(incoming_data_path,"st_lucia_energy","66Poles.shp"))
    poles.drop(["NORTHING","EASTING","Lat","Long_"],axis=1,inplace=True)
    poles["asset_type"] = "pole"

    lines = gpd.read_file(os.path.join(incoming_data_path,"st_lucia_energy","TransOHL.shp"))

    ln = lines.copy()
    ln["geometry"] = ln.geometry.buffer(1)
    lines_poles_match = gpd.sjoin(ln,
                            poles, 
                            how="left", 
                            predicate='intersects'
                        ).reset_index()
                       
    del ln
    poles_matches = list(set(lines_poles_match["POLE_LABEL"].values.tolist()))

    poles = poles[poles["POLE_LABEL"].isin(poles_matches)]

    # Create network topology
    network = create_network_from_nodes_and_edges(
        poles,
        lines,
        "elec"
    )

    network.edges = network.edges.set_crs(epsg=32620)
    network.nodes = network.nodes.set_crs(epsg=32620)

    edges = network.edges
    nodes = network.nodes
    edges['length_m'] = edges.geometry.length
    

    topo_nodes = list(set(edges.from_node.values.tolist() + edges.to_node.values.tolist()))
    nodes = nodes[nodes.node_id.isin(topo_nodes)]

    nodes["POLE_LABEL"] = nodes["POLE_LABEL"].astype(str)
    nodes["asset_type"] = np.where(nodes["asset_type"] == "pole","pole","dummy")

    """ To check if some poles were not included. Not needed
    blank_nodes = nodes[nodes["POLE_LABEL"] == 'nan']
    rem_poles = nodes[nodes["POLE_LABEL"] != 'nan']["POLE_LABEL"].values.tolist()
    rem_poles = poles[~poles["POLE_LABEL"].isin(rem_poles)]
    if len(rem_poles.index) > 0:
        nodes_poles = ckdnearest(rem_poles,blank_nodes[["node_id","geometry"]])
        print (nodes_poles.sort_values(by="dist",ascending=False))

        # nodes_poles.to_file(os.path.join(incoming_data_path,"st_lucia_energy","lca_energy.gpkg"),
        #     layer="poles",
        #     driver="GPKG")
    """

    # Add components
    edges, nodes = components(edges,nodes,'node_id')

    max_edge_id = max([int(v.split("_")[1]) for v in edges["edge_id"].values.tolist()])
    max_node_id = max([int(v.split("_")[1]) for v in nodes["node_id"].values.tolist()])
    network_components = list(set(nodes.component.values.tolist()))
    distances = []
    for i in range(len(network_components)-1):
        component_one = nodes[nodes.component == network_components[i]]
        component_two = nodes[nodes.component.isin(network_components[i+1:])]
        component_distances = ckdnearest(component_one[["node_id","geometry"]],
                                           component_two[["node_id","geometry"]])
        distances.append(component_distances[component_distances["dist"] <= 1.5])

    distances = pd.concat(distances,axis=0,ignore_index=True)
    distances.columns = ["from_node","from_geometry","node_id","length_m"]

    distances = add_new_edges(distances,edges,nodes,max_edge_id)
    
    edges = pd.concat([edges,distances],axis=0,ignore_index=True)
    max_edge_id = max([int(v.split("_")[1]) for v in edges["edge_id"].values.tolist()])
    edges, nodes = components(edges,nodes,'node_id')

    nodes = network_degree(edges,nodes,"node_id")
    
    plants_substations["node_id"] = [f"elecn_{max_node_id + 1 + x}" for x in plants_substations.index.values]
    plants_substations["supply_capacity_MW"] = 0
    plants_substations["supply_capacity_GWH"] = 0
    plants_substations.loc[plants_substations["asset_type"]=="diesel","supply_capacity_MW"] = 88.4
    plants_substations.loc[plants_substations["asset_type"]=="diesel","supply_capacity_GWH"] = 88.4*1000*24*365*0.79/1000000.0
    solar_area_total = sum(plants_substations[plants_substations["asset_type"]=="solar"]["geometry"].area)
    plants_substations.loc[
        plants_substations["asset_type"]=="solar",
        "supply_capacity_MW"] = 3.0*(plants_substations.loc[
                            plants_substations["asset_type"]=="solar",
                            "geometry"].area)/solar_area_total
    plants_substations.loc[
        plants_substations["asset_type"]=="solar",
        "supply_capacity_GWH"] = 7.0*(plants_substations.loc[
                            plants_substations["asset_type"]=="solar",
                            "geometry"].area)/solar_area_total 

    plants_nodes_join = gpd.sjoin(plants_substations[["node_id","geometry"]],
                            nodes[nodes["degree"] == 1][["node_id","geometry"]],
                            how="left", 
                            predicate='intersects'
                        ).reset_index()
    plants_nodes_join.drop(["index","index_right"],axis=1,inplace=True)
    plants_nodes_join.columns = ["from_node","from_geometry","node_id"]
    plants_nodes_join["node_id"] = plants_nodes_join["node_id"].astype(str)
    plants_nodes_join["from_geometry"] = plants_nodes_join.progress_apply(lambda x:x.from_geometry.centroid,axis=1)
    plants_nodes_join = plants_nodes_join[plants_nodes_join["node_id"] != 'nan']
    distances = add_new_edges(plants_nodes_join,edges,nodes,max_edge_id)
    edges = pd.concat([edges,distances],axis=0,ignore_index=True)

    edges, nodes = components(edges,nodes,'node_id')

    gpd.GeoDataFrame(edges,
            geometry="geometry",
            crs="EPSG:32620").to_file(os.path.join(processed_data_path,
                                    "infrastructure/energy",
                                    "lca_energy.gpkg"),
                                    layer="edges",
                                    driver="GPKG")
    gpd.GeoDataFrame(nodes,
        geometry="geometry",
        crs="EPSG:32620").to_file(os.path.join(processed_data_path,
                                "infrastructure/energy",
                                "lca_energy.gpkg"),
                                layer="nodes",
                                driver="GPKG")
    gpd.GeoDataFrame(plants_substations,
        geometry="geometry",
        crs="EPSG:32620").to_file(os.path.join(processed_data_path,
                                "infrastructure/energy",
                                "lca_energy.gpkg"),
                                layer="areas",
                                driver="GPKG")

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
