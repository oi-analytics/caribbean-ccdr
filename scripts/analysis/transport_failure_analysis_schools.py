
import os
os.environ['USE_PYGEOS'] = '0'
from os.path import join
import numpy as np
import networkx as nx
from importlib import reload
import pandas as pd
import geopandas as gpd
from geospatial_utils import load_config
from num2words import num2words
import transport_flow_and_disruption_functions as tfdf
from shapely.geometry import Point
import matplotlib.pyplot as plt
from shapely.geometry import Point
from analysis_utils import get_nearest_values
import matplotlib.pyplot as plt

import matplotlib.colors as colors
plot_kwargs = {'dpi': 400, 'bbox_inches': 'tight'}

caribbean_epsg = 32620

# settings
C = 5
COST = 'time_m'
ZETA = 1
EDGE_ATTRS = ['edge_id', 'length_m', 'time_m']
RECALCULATE_PATHS = False
COUNTRY = 'LCA'


def main(CONFIG):
    # set up directories
    indir, outdir, figdir = CONFIG['paths']['incoming_data'], CONFIG['paths']['data'], CONFIG['paths']['figures']

    # load roads, roads network, and schools
    roads, road_net = tfdf.get_roads(join(outdir, 'infrastructure', 'transport'), COUNTRY, ['edge_id', 'length_m', 'time_m'])
    roads = roads.to_crs(epsg=caribbean_epsg)
    schools = gpd.read_file(join(outdir, 'infrastructure', 'social', 'education.gpkg'))
    schools = schools[schools['iso_code'] == COUNTRY]

    # load school districts (add to this when more countries)
    if COUNTRY == 'LCA':
        admin_areas = gpd.read_file(join(outdir, 'infrastructure', 'social', f'{COUNTRY.lower()}_edu_districts.gpkg'), layer='areas')[["DIS","geometry"]]
        admin_areas["DIS"] = admin_areas.apply(lambda x: str(num2words(x["DIS"])).title(), axis=1)
        admin_areas.rename(columns={"DIS": "school_district"},inplace=True)
        
    admin_areas = admin_areas.to_crs(epsg=caribbean_epsg)
    admin_areas["school_district"] = admin_areas["school_district"].astype(str).str.replace("Saint","St.")

    # divide road dataframe into intra-admin area roads
    roads_by_district = gpd.sjoin(roads, admin_areas, how="inner", predicate='within').reset_index(drop=True).drop('index_right', axis=1)

    paths_df = []
    road_nets = []
    pathname = join(outdir, f'{COUNTRY}_schools_pathdata_{COST}_{C}')
    # process road network and recalculate paths (if RECALCULATE_PATHS)
    for district in [*roads_by_district['school_district'].unique()]:
        # subset geodataframes to district
        roads_district = roads_by_district[roads_by_district['school_district'] == district]
        road_net_dist = road_net.edge_subgraph([(u, v) for u, v in zip(roads_district['from_node'], roads_district['to_node'])])
        schools_district = schools[schools['school_district'] == district]

        # get all node geometries
        node_geoms = {node: Point(line.coords[-1]) for node, line in zip(roads_district['to_node'], roads_district['geometry'])}
        node_geoms = node_geoms | {node: Point(line.coords[0]) for node, line in zip(roads_district['from_node'], roads_district['geometry']) if node not in node_geoms.keys()}
        nodes_gdf = gpd.GeoDataFrame.from_dict(node_geoms, orient='index').reset_index().rename(columns={'index': 'node', 0: 'geometry'}).set_geometry('geometry')

        # find nearest nodes to each school building
        schools_district['nearest_node'] = schools_district.apply(lambda row: get_nearest_values(row, nodes_gdf, 'node'), axis=1)
        aggfunc = {'node_id': lambda x : '_and_'.join(x), 'assigned_students': sum}
        nearest_nodes = schools_district[['nearest_node', 'node_id', 'assigned_students']].groupby('nearest_node').agg(aggfunc).reset_index()

        # replace nodes with schools in road network
        rename_schools = {node_id: school_id for node_id, school_id in zip(nearest_nodes['nearest_node'], nearest_nodes['node_id'])}
        school_pops = {school_id: school_pop for school_id, school_pop in zip(nearest_nodes['node_id'], nearest_nodes['assigned_students'])}
        school_classes = {school_id: "school" for school_id in nearest_nodes['node_id']}
        road_net_dist = nx.relabel_nodes(road_net_dist, rename_schools, copy=True)
        nx.set_node_attributes(road_net_dist, school_pops, name="population")
        nx.set_node_attributes(road_net_dist, "domestic", name="class")
        nx.set_node_attributes(road_net_dist, school_classes, name="class")

        # some tests here
        tfdf.test_pop_assignment(road_net_dist, nearest_nodes, rename_schools)
        # test_plot(nodes_gdf, nearest_nodes)

        # get path and flux data
        if RECALCULATE_PATHS:
            path_df = tfdf.get_flux_data(road_net_dist, COST, C, ZETA, origin_class='domestic', dest_class='school', class_str='class')
            path_df.loc[:, 'school_district'] = district
            paths_df.append(path_df)

        road_nets.append(road_net)

    if RECALCULATE_PATHS:
        pd.concat(paths_df).to_parquet(path=f"{pathname}.parquet", index=True)
            
    school_road_net = nx.compose_all(road_nets)
    paths_df = pd.read_parquet(path=f"{pathname}.parquet", engine="fastparquet")

    # add traffic to road edges
    pathname = join(outdir, f"{COUNTRY}_schools_traffic_{COST}_{C}")
    traffic_df = tfdf.get_flow_on_edges(paths_df, 'edge_id', 'edge_path', 'flux')
    traffic_df = traffic_df.rename(columns={'flux': 'traffic'})
    roads_traffic = roads_by_district.merge(traffic_df, how='left', on='edge_id')
    roads_traffic.to_file(filename=f"{pathname}.gpkg", driver="GPKG", layer="roads")

    fig = tfdf.plot_district_traffic(roads_traffic, schools, 'Two', "school_district", "assigned_students")
    fig.savefig(join(figdir, f"{COUNTRY}_schools_traffic_zoom_{COST}_{C}.png"), **plot_kwargs)

    # load data fresh (optional, from notebook)
    paths_df = pd.read_parquet(join(outdir, f'{COUNTRY}_schools_pathdata_{COST}_{C}.parquet'), engine="fastparquet")
    roads_traffic = gpd.read_file(join(outdir, f"{COUNTRY}_schools_traffic_{COST}_{C}.gpkg"))
    road_net = school_road_net.copy()
        
    # set up simulated disruption
    scenario_dict = tfdf.simulate_disruption(roads_traffic, seed=2)
    path_df_disrupted = tfdf.model_disruption(scenario_dict, paths_df, school_road_net, outdir, COUNTRY, COST, C)
    roads_disrupted = tfdf.get_traffic_disruption(paths_df, path_df_disrupted, roads_traffic, scenario_dict, COST)

    # plot changes to traffic
    fig = tfdf.plot_failed_edge_traffic(roads_disrupted, scenario_dict, COUNTRY, COST, C)
    fig.savefig(join(figdir, f"{COUNTRY}_schools_traffic_change_zoom_{scenario_dict['scenario_name']}_{COST}_{C}.png"), **plot_kwargs)


if __name__ == "__main__":
    CONFIG = load_config(join("..", "..", ".."))
    main(CONFIG)