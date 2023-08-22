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

def new_edge_properties(new_edges,max_edge_id):
    new_edges["edge_id"] = list(max_edge_id + 1 + new_edges.index.values)
    new_edges["edge_id"] = new_edges.progress_apply(lambda x: f"elece_{x.edge_id}",axis=1)
    new_edges["length_m"] = new_edges.geometry.length
    new_edges["feedername"] = "dummy"
    new_edges["asset_type"] = "dummy"
    new_edges["line_type"] = "dummy"
    new_edges["powerlinet"] = "dummy"
    new_edges["voltage"] = "unknown"
    # new_edges["shape_leng"] = new_edges["length_m"]
    # new_edges["objectid"] = list(max(existing_edges.objectid.values.tolist()) + 1 + new_edges.index.values)
    # new_edges = gpd.GeoDataFrame(new_edges,geometry="geometry",crs=f"EPSG:{epsg}")
    return new_edges

def add_new_edges(new_edges,existing_nodes,max_edge_id,epsg=32620):
    new_edges = pd.merge(new_edges,existing_nodes[["node_id","geometry"]],how="left",on=["node_id"])
    new_edges.rename(columns={"node_id":"to_node","geometry":"to_geometry"},inplace=True)
    new_edges["geometry"] = new_edges.progress_apply(lambda x:LineString([x.from_geometry,x.to_geometry]),axis=1)
    new_edges.drop(["from_geometry","to_geometry"],axis=1,inplace=True)
    new_edges = gpd.GeoDataFrame(new_edges,geometry="geometry",crs=f"EPSG:{epsg}")
    new_edges = new_edge_properties(new_edges,max_edge_id)
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

def assign_node_function(x,source_types,sink_types):
    if str(x.asset_type).lower() in source_types:
        return "source"
    elif str(x.asset_type).lower() in sink_types:
        return "sink"
    else:
        return "intermediate"

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    epsg_caribbean = 32620

    """
    Written to preproces plants data. Not needed anymore
    plants_substations = []
    for n in ['','_1','_2','_3','_4']:
        plants_substations.append(
                    gpd.read_file(
                        os.path.join(
                            incoming_data_path,
                            "grd_vct_power",
                            f"vct_energy_osm_extract{n}.gpkg"
                        )
                    )[["osm_id","geometry"]])
    plants_substations = pd.concat(plants_substations,axis=0,ignore_index=True)
    plants_substations = gpd.GeoDataFrame(plants_substations,geometry="geometry",crs="EPSG:4326")
    plants_substations = plants_substations.to_crs(epsg=epsg_caribbean)
    print (plants_substations)
    plants_substations.to_file(os.path.join(
                            incoming_data_path,
                            "grd_vct_power",
                            f"vct_energy_osm_extracts.gpkg"
                            ))
    """
    input_data_path = os.path.join(incoming_data_path,"SVG/GIS DATA_March_2018")
    poles = gpd.read_file(os.path.join(input_data_path,"Pole_March27_2018.shp"))
    poles = poles.to_crs(epsg=epsg_caribbean)
    poles["asset_type"] = "pole"

    poles.to_file(os.path.join(input_data_path,"vct_grid.gpkg"),layer="nodes",driver="GPKG")
    lines = []
    line_files = ["OHPrimaryLines","OHSecondaryLines",
                    "OHTransmissionLines","UGPrimaryLines",
                    "UGSecondaryLines","UGTransmissionLines"]
    line_names = ["ohp","ohs","oht","ugp","ugs","ugt"]
    line_columns = ["powerlinet","voltage","feedername"]
    for idx,(lf,ln) in enumerate(zip(line_files,line_names)):
        l_df = gpd.read_file(os.path.join(input_data_path,f"{lf}.shp")) 
        l_df = l_df.to_crs(epsg=epsg_caribbean)
        l_df.columns = map(str.lower, l_df.columns)
        l_df["line_id"] = l_df.index.values.tolist()
        l_df["line_id"] = l_df.progress_apply(lambda x: f"{ln}_{x.line_id}",axis=1)
        l_df["line_type"] = ln 
        if ln in ["ohp","ohs","oht"]:
            l_df["asset_type"] = "ohl"
        else:
            l_df["asset_type"] = "uc"
        lines.append(l_df[["line_id","line_type","asset_type","geometry"] + line_columns])

    lines = gpd.GeoDataFrame(
                    pd.concat(lines,axis=0,ignore_index=True),
                    geometry="geometry",crs=f"EPSG:{epsg_caribbean}")
    lines.to_file(os.path.join(input_data_path,"vct_grid.gpkg"),layer="edges",driver="GPKG")

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

    distances = add_new_edges(distances,nodes,max_edge_id)
    
    edges = pd.concat([edges,distances],axis=0,ignore_index=True)
    max_edge_id = max([int(v.split("_")[1]) for v in edges["edge_id"].values.tolist()])
    edges, nodes = components(edges,nodes,'node_id')
    nodes = network_degree(edges,nodes,"node_id")

    # store_imtermediate = True
    # if store_imtermediate is True:
    #     gpd.GeoDataFrame(edges,
    #         geometry="geometry",
    #         crs="EPSG:32620").to_file(os.path.join(processed_data_path,
    #                                 "infrastructure/energy",
    #                                 "vct_energy.gpkg"),
    #                                 layer="edges",
    #                                 driver="GPKG")
    #     gpd.GeoDataFrame(nodes,
    #         geometry="geometry",
    #         crs="EPSG:32620").to_file(os.path.join(processed_data_path,
    #                                 "infrastructure/energy",
    #                                 "vct_energy.gpkg"),
    #                                 layer="nodes",
    #                                 driver="GPKG")

    # edges = gpd.read_file(os.path.join(processed_data_path,
    #                                 "infrastructure/energy",
    #                                 "vct_energy.gpkg"),
    #                                 layer="edges")
    # nodes = gpd.read_file(os.path.join(processed_data_path,
    #                                 "infrastructure/energy",
    #                                 "vct_energy.gpkg"),
    #                                 layer="nodes")
    max_edge_id = max([int(v.split("_")[1]) for v in edges["edge_id"].values.tolist()])
    max_node_id = max([int(v.split("_")[1]) for v in nodes["node_id"].values.tolist()])
    plants_substations = gpd.read_file(os.path.join(
                                            incoming_data_path,
                                            "grd_vct_power",
                                            f"vct_energy_osm_extracts.gpkg"
                                            ))
    plants_substations["node_id"] = [f"elecn_{max_node_id + 1 + x}" for x in plants_substations.index.values]
    plants = plants_substations[plants_substations["asset_type"] != "substation"]
    plants["geometry"] = plants.geometry.centroid
    substations = plants_substations[plants_substations["asset_type"] == "substation"]
    substations["geometry"] = substations.geometry.centroid

    distances = ckdnearest(plants[["node_id","geometry"]],
                           substations[["node_id","geometry"]])
    distances.columns = ["from_node","from_geometry","node_id","length_m"]
    unmatched_plants = list(set(distances[distances["length_m"] > 150.0]["from_node"].values.tolist()))
    print (distances)
    
    distances = distances[distances["length_m"] <= 150.0]
    # distances.columns = ["from_node","from_geometry","node_id","length_m"]
    print (distances)
    distances = add_new_edges(distances,substations,max_edge_id)
    edges = pd.concat([edges,distances],axis=0,ignore_index=True)
    
    max_edge_id = max([int(v.split("_")[1]) for v in edges["edge_id"].values.tolist()])
    
    plants = plants_substations[
                        (
                            plants_substations["asset_type"] == "substation"
                            ) | (
                            plants_substations["node_id"].isin(unmatched_plants)
                            )
                        ]   
    plants["geometry"] = plants.geometry.buffer(200)

    select_edges = edges[edges["line_type"].isin(["ohp","oht","ugp","ugt"])]
    select_nodes = list(set(select_edges["from_node"].values.tolist() + select_edges["to_node"].values.tolist()))
    plants_nodes_join = gpd.sjoin(plants[["node_id","geometry"]],
                            nodes[(nodes["degree"] == 1) & (nodes["node_id"].isin(select_nodes))][["node_id","geometry"]],
                            how="left", 
                            predicate='intersects'
                        ).reset_index()
    plants_nodes_join.drop(["index","index_right"],axis=1,inplace=True)
    plants_nodes_join.columns = ["from_node","from_geometry","node_id"]
    plants_nodes_join["node_id"] = plants_nodes_join["node_id"].astype(str)
    plants_nodes_join["from_geometry"] = plants_nodes_join.progress_apply(lambda x:x.from_geometry.centroid,axis=1)
    plants_nodes_join = plants_nodes_join[plants_nodes_join["node_id"] != 'nan']
    distances = add_new_edges(plants_nodes_join,nodes,max_edge_id)
    edges = pd.concat([edges,distances],axis=0,ignore_index=True)

    max_edge_id = max([int(v.split("_")[1]) for v in edges["edge_id"].values.tolist()])
    special_node = "elecn_75442"
    distances = ckdnearest(nodes[nodes["node_id"] == special_node][["node_id","geometry"]],
                           substations[["node_id","geometry"]])
    distances.columns = ["from_node","from_geometry","node_id","length_m"]
    distances = add_new_edges(distances,substations,max_edge_id)
    edges = pd.concat([edges,distances],axis=0,ignore_index=True)
    

    edges, nodes = components(edges,nodes,'node_id')
    nodes = network_degree(edges,nodes,"node_id")
    
    nodes["function_type"] = np.where(nodes["degree"] == 1, "sink","intermediate")

    gpd.GeoDataFrame(plants_substations,
            geometry="geometry",
            crs="EPSG:32620").to_file(os.path.join(processed_data_path,
                                    "infrastructure/energy",
                                    "vct_energy.gpkg"),
                                    layer="areas",
                                    driver="GPKG")

    gpd.GeoDataFrame(edges,
            geometry="geometry",
            crs="EPSG:32620").to_file(os.path.join(processed_data_path,
                                    "infrastructure/energy",
                                    "vct_energy.gpkg"),
                                    layer="edges",
                                    driver="GPKG")
    gpd.GeoDataFrame(nodes,
        geometry="geometry",
        crs="EPSG:32620").to_file(os.path.join(processed_data_path,
                                "infrastructure/energy",
                                "vct_energy.gpkg"),
                                layer="nodes",
                                driver="GPKG")



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
