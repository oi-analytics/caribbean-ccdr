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
from shapely.ops import linemerge
from shapely.geometry import Point,LineString,MultiLineString
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
    # G.add_nodes_from(
    #     (getattr(n, node_id_col), {"geometry": n.geometry}) for n in nodes.itertuples()
    # )
    G.add_edges_from(
        (e.from_node, e.to_node, {"edge_id": e.edge_id, "geometry": e.geometry})
        for e in edges.itertuples()
    )
    degrees = pd.DataFrame([(node,val) for (node, val) in G.degree()],columns=[node_id_col,"degree"])
    if "degree" in nodes.columns.values.tolist():
        nodes.drop("degree",axis=1,inplace=True)
    nodes = pd.merge(nodes,degrees,how="left",on=[node_id_col])

    return nodes

def new_edge_properties(new_edges,existing_edges,max_edge_id):
    new_edges["edge_id"] = list(max_edge_id + 1 + new_edges.index.values)
    new_edges["edge_id"] = new_edges.progress_apply(lambda x: f"elece_{x.edge_id}",axis=1)
    new_edges["length_m"] = new_edges.geometry.length
    new_edges["feedernm"] = "dummy"
    new_edges["asset_type"] = "dummy"
    new_edges["shape_leng"] = new_edges["length_m"]
    new_edges["objectid"] = list(max(existing_edges.objectid.values.tolist()) + 1 + new_edges.index.values)
    # new_edges = gpd.GeoDataFrame(new_edges,geometry="geometry",crs=f"EPSG:{epsg}")
    return new_edges

def add_new_edges(new_edges,existing_edges,existing_nodes,max_edge_id,epsg=32620):
    new_edges = pd.merge(new_edges,existing_nodes[["node_id","geometry"]],how="left",on=["node_id"])
    new_edges.rename(columns={"node_id":"to_node","geometry":"to_geometry"},inplace=True)
    new_edges["geometry"] = new_edges.progress_apply(lambda x:LineString([x.from_geometry,x.to_geometry]),axis=1)
    new_edges.drop(["from_geometry","to_geometry"],axis=1,inplace=True)
    new_edges = gpd.GeoDataFrame(new_edges,geometry="geometry",crs=f"EPSG:{epsg}")
    new_edges = new_edge_properties(new_edges,existing_edges,max_edge_id)
    return new_edges

def get_od_nodes(nd,ty,plants,nodes):
    if ty == "name":
        df = plants
    else:
        df = nodes
    return df[df[ty] == nd][["node_id","geometry"]]

def closest_road_straightline(energy_node,road_nodes):
    closest_road = ckdnearest(energy_node,
                            road_nodes[["road_id","geometry"]])

    st_line_geom = LineString([
                energy_node.geometry.values[0],
                road_nodes[road_nodes["road_id"] == closest_road["road_id"].values[0]
                ].geometry.values[0]]
                )
    return closest_road, st_line_geom

def merge_lines(single_linestrings):
    inlines = MultiLineString(single_linestrings)
    # Put the sub-line coordinates into a list of sublists
    # outcoords = [list(i.coords) for i in inlines]
    # Flatten the list of sublists and use it to make a new line
    # outline = LineString([i for sublist in outcoords for i in sublist])

    outline = linemerge(inlines)

    return outline

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']

    plants_substations = gpd.read_file(os.path.join(incoming_data_path,"dominica_electricalnet","electricPS.shp"))
    plants_substations.columns = map(str.lower, plants_substations.columns)
    plants_substations.rename(columns={"type":"asset_type"},inplace=True)
    plants_substations = plants_substations.explode(index_parts=False)
    plants_substations = plants_substations.to_crs(epsg=32620)

    lines = gpd.read_file(os.path.join(incoming_data_path,"dominica_electricalnet","electricNet.shp"))
    lines.columns = map(str.lower, lines.columns)
    lines["asset_type"] = "OHL"
    lines = lines.to_crs(epsg=32620)

    # Create network topology
    network = create_network_from_nodes_and_edges(
        plants_substations,
        lines,
        "elec",
        snap_distance=400
    )

    network.edges = network.edges.set_crs(epsg=32620)
    network.nodes = network.nodes.set_crs(epsg=32620)

    edges = network.edges
    nodes = network.nodes
    edges['length_m'] = edges.geometry.length

    edges, nodes = components(edges,nodes,'node_id')
    nodes = network_degree(edges,nodes,"node_id")

    edges["asset_type"] = edges["asset_type"].astype(str).str.replace("nan","dummy")
    nodes["asset_type"] = nodes["asset_type"].astype(str).str.replace("nan","dummy")
    gpd.GeoDataFrame(edges,
            geometry="geometry",
            crs="EPSG:32620").to_file(os.path.join(processed_data_path,
                                    "infrastructure/energy",
                                    "dma_energy.gpkg"),
                                    layer="edges",
                                    driver="GPKG")
    gpd.GeoDataFrame(nodes,
        geometry="geometry",
        crs="EPSG:32620").to_file(os.path.join(processed_data_path,
                                "infrastructure/energy",
                                "dma_energy.gpkg"),
                                layer="nodes",
                                driver="GPKG")

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
