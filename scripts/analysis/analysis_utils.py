"""Functions for preprocessing data
"""
import sys
import os
import json

import pandas as pd
import geopandas as gpd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, shape
from scipy.interpolate import interp1d
from collections import defaultdict
from itertools import chain
from scipy import integrate
from scipy.spatial import cKDTree
import igraph as ig
import fiona
import math
import numpy as np
from tqdm import tqdm
tqdm.pandas()

def load_config():
    """Read config.json"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
    with open(config_path, "r") as config_fh:
        config = json.load(config_fh)
    return config

def modify_epoch(hazard_dataframe,baseline_year):
    if "epoch" not in hazard_dataframe.columns.values.tolist():
        hazard_dataframe["epoch"] = baseline_year

    hazard_dataframe["epoch"] = hazard_dataframe["epoch"].fillna(baseline_year)
    hazard_dataframe.loc[hazard_dataframe["epoch"] < baseline_year,"epoch"] = baseline_year
    return hazard_dataframe

def gdf_geom_clip(gdf_in, clip_geom):
    """Filter a dataframe to contain only features within a clipping geometry

    Parameters
    ---------
    gdf_in
        geopandas dataframe to be clipped in
    province_geom
        shapely geometry of province for what we do the calculation

    Returns
    -------
    filtered dataframe
    """
    return gdf_in.loc[gdf_in['geometry'].apply(lambda x: x.within(clip_geom))].reset_index(drop=True)

def get_nearest_values(x,input_gdf,column_name):
    polygon_index = input_gdf.distance(x.geometry).sort_values().index[0]
    return input_gdf.loc[polygon_index,column_name]

def extract_gdf_values_containing_nodes(x, input_gdf, column_name):
    a = input_gdf.loc[list(input_gdf.geometry.contains(x.geometry))]
    if len(a.index) > 0:
        return a[column_name].values[0]
    else:
        polygon_index = input_gdf.distance(x.geometry).sort_values().index[0]
        return input_gdf.loc[polygon_index,column_name]

def nearest(geom, gdf):
    """Find the element of a GeoDataFrame nearest a shapely geometry
    """
    matches_idx = gdf.sindex.nearest(geom.bounds)
    nearest_geom = min(
        [gdf.iloc[match_idx] for match_idx in matches_idx],
        key=lambda match: geom.distance(match.geometry)
    )
    return nearest_geom

def get_nearest_node(x, sindex_input_nodes, input_nodes, id_column):
    """Get nearest node in a dataframe

    Parameters
    ----------
    x
        row of dataframe
    sindex_nodes
        spatial index of dataframe of nodes in the network
    nodes
        dataframe of nodes in the network
    id_column
        name of column of id of closest node

    Returns
    -------
    Nearest node to geometry of row
    """
    return input_nodes.loc[list(sindex_input_nodes.nearest(x.bounds[:2]))][id_column].values[0]


def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite voronoi regions in a 2D diagram to finite regions.

    Source: https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram-python

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def assign_value_in_area_proportions(poly_1_gpd, poly_2_gpd, poly_attribute):
    poly_1_sindex = poly_1_gpd.sindex
    for p_2_index, polys_2 in poly_2_gpd.iterrows():
        poly2_attr = 0
        intersected_polys = poly_1_gpd.iloc[list(
            poly_1_sindex.intersection(polys_2.geometry.bounds))]
        for p_1_index, polys_1 in intersected_polys.iterrows():
            if (polys_2['geometry'].intersects(polys_1['geometry']) is True) and (polys_1.geometry.is_valid is True) and (polys_2.geometry.is_valid is True):
                poly2_attr += polys_1[poly_attribute]*polys_2['geometry'].intersection(
                    polys_1['geometry']).area/polys_1['geometry'].area

        poly_2_gpd.loc[p_2_index, poly_attribute] = poly2_attr

    return poly_2_gpd

def extract_nodes_within_gdf(x, input_nodes, column_name):
    a = input_nodes.loc[list(input_nodes.geometry.within(x.geometry))]
    # if len(a.index) > 1: # To check if there are multiple intersections
    #     print (x)
    if len(a.index) > 0:
        return a[column_name].values[0]
    else:
        return ''

def create_voronoi_polygons_from_nodes(nodes_dataframe,bounding_dataframe,node_id_column,epsg=4326,**kwargs):
    # create Voronoi polygons for the nodes
    nodes_dataframe = nodes_dataframe.reset_index()
    xy_list = []
    for iter_, values in nodes_dataframe.iterrows():
        xy = list(values.geometry.coords)
        xy_list += [list(xy[0])]

    vor = Voronoi(np.array(xy_list))
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # min_x = vor.min_bound[0] - 0.1
    # max_x = vor.max_bound[0] + 0.1
    # min_y = vor.min_bound[1] - 0.1
    # max_y = vor.max_bound[1] + 0.1

    # min_x = vor.min_bound[0]
    # max_x = vor.max_bound[0]
    # min_y = vor.min_bound[1]
    # max_y = vor.max_bound[1]

    # mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
    # bounded_vertices = np.max((vertices, mins), axis=0)
    # maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
    # bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

    # box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    poly_list = []
    for region in regions:
        polygon = vertices[region]
        # Clipping polygon
        poly = Polygon(polygon)
        # poly = poly.intersection(box)
        poly_list.append(poly)

    poly_index = list(np.arange(0, len(poly_list), 1))
    poly_df = pd.DataFrame(list(zip(nodes_dataframe[node_id_column].values.tolist(), poly_list)),
                                   columns=[node_id_column, 'geometry'])
    gdf_voronoi = gpd.GeoDataFrame(poly_df, geometry = 'geometry',crs=f'epsg:{epsg}')
    gdf_voronoi = gpd.clip(gdf_voronoi, bounding_dataframe)
    gdf_voronoi['areas'] = gdf_voronoi.progress_apply(lambda x:x.geometry.area,axis=1)
    # gdf_voronoi[node_id_column] = gdf_voronoi.progress_apply(
    #     lambda x: extract_nodes_within_gdf(x, nodes_dataframe, node_id_column), axis=1)
    if not kwargs.get('save',False):
        pass
    else:
        gdf_voronoi.to_file(kwargs.get('voronoi_path','voronoi-output.shp'))

    return gdf_voronoi

def nodes_voronoi_creations(nodes_df,node_id_column,boundary_df,epsg=4326):
    nodes_voronoi = create_voronoi_polygons_from_nodes(
                            nodes_df,
                            boundary_df,
                            node_id_column,
                            epsg=epsg)
    # nodes_voronoi = gpd.clip(nodes_voronoi, boundary_df)
    return nodes_voronoi

def assign_weights_by_area_intersections(gdf_voronoi,
                        population_dataframe,
                        node_id_column,population_value_column):
    """Assign weights to nodes based on their nearest populations

        - By finding the population that intersect with the Voronoi extents of nodes

    Parameters
        - nodes_dataframe - Geodataframe of the nodes
        - population_dataframe - Geodataframe of the population
        - nodes_id_column - String name of node ID column
        - population_value_column - String name of column containing population values

    Outputs
        - nodes - Geopandas dataframe of nodes with new column called population
    """

    # load provinces and get geometry of the right population_dataframe
    sindex_population_dataframe = population_dataframe.sindex

    gdf_voronoi[population_value_column] = 0
    gdf_voronoi = assign_value_in_area_proportions(population_dataframe, gdf_voronoi, population_value_column)
    gdf_voronoi = gdf_voronoi[~(gdf_voronoi[node_id_column] == '')]

    return gdf_voronoi[[node_id_column, population_value_column]]
    

def ckdnearest(gdA, gdB):
    """Taken from https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    """
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

# def drop_duplicate_geometries(gdf, keep="first"):
#     """Drop duplicate geometries from a dataframe"""
#     # convert to wkb so drop_duplicates will work
#     # discussed in https://github.com/geopandas/geopandas/issues/521
#     mask = gdf.geometry.apply(lambda geom: geom.wkb)
#     # use dropped duplicates index to drop from actual dataframe
#     return gdf.iloc[mask.drop_duplicates(keep).index]

# def split_multigeometry(dataframe,split_geometry_type="GeometryCollection"):
#     """Create multiple geometries from any MultiGeomtery and GeometryCollection

#     Ensures that edge geometries are all Points,LineStrings,Polygons, duplicates attributes over any
#     created multi-geomteries.
#     """
#     simple_geom_attrs = []
#     simple_geom_geoms = []
#     for v in tqdm(dataframe.itertuples(index=False),
#                      desc="split_multi",
#                      total=len(dataframe)):
#         if v.geometry.geom_type == split_geometry_type:
#             geom_parts = list(v.geometry)
#         else:
#             geom_parts = [v.geometry]

#         for part in geom_parts:
#             simple_geom_geoms.append(part)

#         attrs = gpd.GeoDataFrame([v] * len(geom_parts))
#         simple_geom_attrs.append(attrs)

#     simple_geom_geoms = gpd.GeoDataFrame(simple_geom_geoms, columns=["geometry"])
#     dataframe = (pd.concat(simple_geom_attrs,
#                            axis=0).reset_index(drop=True).drop("geometry",
#                                                                axis=1))
#     dataframe = pd.concat([dataframe, simple_geom_geoms], axis=1)

#     return dataframe

def get_flow_on_edges(save_paths_df,edge_id_column,edge_path_column,
    flow_column):
    """Write results to Shapefiles

    Outputs ``gdf_edges`` - a shapefile with minimum and maximum tonnage flows of all
    commodities/industries for each edge of network.

    Parameters
    ---------
    save_paths_df
        Pandas DataFrame of OD flow paths and their tonnages
    industry_columns
        List of string names of all OD commodities/industries indentified
    min_max_exist
        List of string names of commodity/industry columns for which min-max tonnage column names already exist
    gdf_edges
        GeoDataFrame of network edge set
    save_csv
        Boolean condition to tell code to save created edge csv file
    save_shapes
        Boolean condition to tell code to save created edge shapefile
    shape_output_path
        Path where the output shapefile will be stored
    csv_output_path
        Path where the output csv file will be stored

    """
    edge_flows = defaultdict(float)
    for row in save_paths_df.itertuples():
        for item in getattr(row,edge_path_column):
            edge_flows[item] += getattr(row,flow_column)

    return pd.DataFrame([(k,v) for k,v in edge_flows.items()],columns=[edge_id_column,flow_column])

def get_flow_paths_indexes_of_edges(flow_dataframe,path_criteria):
    edge_path_index = defaultdict(list)
    for v in flow_dataframe.itertuples():
        for k in getattr(v,path_criteria):
            edge_path_index[k].append(v.Index)

    del flow_dataframe
    return edge_path_index

def get_path_indexes_for_edges(edge_ids_with_paths,selected_edge_list):
    return list(
            set(
                list(
                    chain.from_iterable([
                        path_idx for path_key,path_idx in edge_ids_with_paths.items() if path_key in selected_edge_list
                                        ]
                                        )
                    )
                )
            )

def network_od_path_estimations(graph,
    source, target, cost_criteria):
    """Estimate the paths, distances, times, and costs for given OD pair

    Parameters
    ---------
    graph
        igraph network structure
    source
        String/Float/Integer name of Origin node ID
    source
        String/Float/Integer name of Destination node ID
    tonnage : float
        value of tonnage
    vehicle_weight : float
        unit weight of vehicle
    cost_criteria : str
        name of generalised cost criteria to be used: min_gcost or max_gcost
    time_criteria : str
        name of time criteria to be used: min_time or max_time
    fixed_cost : bool

    Returns
    -------
    edge_path_list : list[list]
        nested lists of Strings/Floats/Integers of edge ID's in routes
    path_dist_list : list[float]
        estimated distances of routes
    path_time_list : list[float]
        estimated times of routes
    path_gcost_list : list[float]
        estimated generalised costs of routes

    """
    paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="epath")


    edge_path_list = []
    path_gcost_list = []
    # for p in range(len(paths)):
    for path in paths:
        edge_path = []
        path_gcost = 0
        if path:
            for n in path:
                edge_path.append(graph.es[n]['edge_id'])
                path_gcost += graph.es[n][cost_criteria]

        edge_path_list.append(edge_path)
        path_gcost_list.append(path_gcost)

    
    return edge_path_list, path_gcost_list

def network_od_paths_assembly(points_dataframe, graph,
                                cost_criteria,store_edge_path=True):
    """Assemble estimates of OD paths, distances, times, costs and tonnages on networks

    Parameters
    ----------
    points_dataframe : pandas.DataFrame
        OD nodes and their tonnages
    graph
        igraph network structure
    region_name : str
        name of Province
    excel_writer
        Name of the excel writer to save Pandas dataframe to Excel file

    Returns
    -------
    save_paths_df : pandas.DataFrame
        - origin - String node ID of Origin
        - destination - String node ID of Destination
        - min_edge_path - List of string of edge ID's for paths with minimum generalised cost flows
        - max_edge_path - List of string of edge ID's for paths with maximum generalised cost flows
        - min_netrev - Float values of estimated netrevenue for paths with minimum generalised cost flows
        - max_netrev - Float values of estimated netrevenue for paths with maximum generalised cost flows
        - min_croptons - Float values of estimated crop tons for paths with minimum generalised cost flows
        - max_croptons - Float values of estimated crop tons for paths with maximum generalised cost flows
        - min_distance - Float values of estimated distance for paths with minimum generalised cost flows
        - max_distance - Float values of estimated distance for paths with maximum generalised cost flows
        - min_time - Float values of estimated time for paths with minimum generalised cost flows
        - max_time - Float values of estimated time for paths with maximum generalised cost flows
        - min_gcost - Float values of estimated generalised cost for paths with minimum generalised cost flows
        - max_gcost - Float values of estimated generalised cost for paths with maximum generalised cost flows

    """
    save_paths = []
    points_dataframe = points_dataframe.set_index('origin_id')
    origins = list(set(points_dataframe.index.values.tolist()))
    for origin in origins:
        try:
            destinations = list(set(points_dataframe.loc[[origin], 'destination_id'].values.tolist()))

            get_path, get_gcost = network_od_path_estimations(
                graph, origin, destinations, cost_criteria)

            # tons = points_dataframe.loc[[origin], tonnage_column].values
            save_paths += list(zip([origin]*len(destinations),
                                destinations, get_path,
                                get_gcost))

            # print(f"done with {origin}")
        except:
            print(f"* no path between {origin}-{destinations}")
    
    cols = [
        'origin_id', 'destination_id', 'edge_path','gcost'
    ]
    save_paths_df = pd.DataFrame(save_paths, columns=cols)
    if store_edge_path is False:
        save_paths_df.drop("edge_path",axis=1,inplace=True)

    points_dataframe = points_dataframe.reset_index()
    # save_paths_df = pd.merge(save_paths_df, points_dataframe, how='left', on=[
    #                          'origin_id', 'destination_id']).fillna(0)

    save_paths_df = pd.merge(points_dataframe,save_paths_df,how='left', on=[
                             'origin_id', 'destination_id']).fillna(0)

    # save_paths_df = save_paths_df[(save_paths_df[tonnage_column] > 0)
    #                               & (save_paths_df['origin_id'] != 0)]
    save_paths_df = save_paths_df[save_paths_df['origin_id'] != 0]

    return save_paths_df

def network_od_path_estimations(graph,
    source, target, cost_criteria):
    """Estimate the paths, distances, times, and costs for given OD pair
    Parameters
    ---------
    graph
        igraph network structure
    source
        String/Float/Integer name of Origin node ID
    target
        String/Float/Integer name of Destination node ID
    cost_criteria : str
        name of generalised cost criteria to be used: min_gcost or max_gcost
    Returns
    -------
    node_path_list : list[list]
        nested lists of Strings/Floats/Integers of edge ID's in routes
    edge_path_list : list[list]
        nested lists of Strings/Floats/Integers of edge ID's in routes
    """
    node_paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="vpath")
    edge_paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="epath")

    node_path_list = []
    for path in node_paths:
        node_path = []
        if path:
            for n in path:
                node_path.append(graph.vs[n]['name'])
        node_path_list.append(node_path)

    edge_path_list = []
    for path in edge_paths:
        edge_path = []
        if path:
            for n in path:
                edge_path.append(graph.es[n]['edge_id'])
        edge_path_list.append(edge_path)

    return node_path_list, edge_path_list

def create_igraph_from_dataframe(graph_dataframe, directed=False, simple=False):
    graph = ig.Graph.TupleList(
        graph_dataframe.itertuples(index=False),
        edge_attrs=list(graph_dataframe.columns)[2:],
        directed=directed
    )
    if simple:
        graph.simplify()

    es, vs, simple = graph.es, graph.vs, graph.is_simple()
    d = "directed" if directed else "undirected"
    s = "simple" if simple else "multi"
    print(
        "Created {}, {} {}: {} edges, {} nodes.".format(
            s, d, "igraph", len(es), len(vs)))

    return graph


def network_ods_assembly(points_dataframe,graph_dataframe,cost_criteria,attribute_columns,directed=True,
                                file_output_path=''):
    """Assemble estimates of OD paths, distances, times, costs and tonnages on networks
    Parameters
    ----------
    points_dataframe : pandas.DataFrame
        OD nodes and their tonnages
    graph
        igraph network structure
    region_name : str
        name of Province
    excel_writer
        Name of the excel writer to save Pandas dataframe to Excel file
    Returns
    -------
    save_paths_df : pandas.DataFrame
        - origin_id - String node ID of Origin
        - destination_id - String node ID of Destination
        - min_edge_path - List of string of edge ID's for paths with minimum generalised cost flows
        - node_path - List of string of node ID's for paths
        - edge_path - List of string of edge ID's for paths
        - flow attributes
    """
    graph = create_igraph_from_dataframe(graph_dataframe,directed=directed)
    save_paths = []
    points_dataframe = points_dataframe.set_index('origin_id')
    origins = list(set(points_dataframe.index.values.tolist()))
    for origin in origins:
        try:
            destinations = points_dataframe.loc[[origin], 'destination_id'].values.tolist()

            get_node_path, get_edge_path = network_od_path_estimations(
                graph, origin, destinations,cost_criteria)

            save_paths += list(zip([origin]*len(destinations), destinations, get_node_path, get_edge_path))

            print(f"done with {origin}")
        except:
            print(f'* no path between {origin}-{destinations}')

    cols = [
        'origin_id', 'destination_id', 'node_path', 'edge_path'
    ]
    if save_paths:
        save_paths_df = pd.DataFrame(save_paths, columns=cols)
        save_paths_df = save_paths_df[save_paths_df.node_path.apply(lambda x: x != [])]
        # print (save_paths_df)
        if attribute_columns:
            points_dataframe = points_dataframe.reset_index()
            save_paths_df = pd.merge(save_paths_df, points_dataframe, how='left', on=[
                                     'origin_id', 'destination_id']).fillna(0)

            save_paths_df = save_paths_df[save_paths_df['origin_id'] != 0]
        if file_output_path:
            save_paths_df.to_parquet(file_output_path, index=False)
        del save_paths

        return save_paths_df
    else:
        return pd.DataFrame()