"""Plot traffic and rerouting for each school district."""
import os
os.environ['USE_PYGEOS'] = '0'
from os.path import join
import networkx as nx
from importlib import reload
import pandas as pd
import geopandas as gpd
from geospatial_utils import load_config
from num2words import num2words
import transport_flow_and_disruption_functions as tfdf
from shapely.geometry import Point
from shapely.geometry import Point
from analysis_utils import get_nearest_values


# global settings
COST = 'time_m'
THRESH = 5
ZETA = 1
EDGE_ATTRS = ['edge_id', 'length_m', 'time_m']
RECALCULATE_PATHS = False
COUNTRY = 'LCA'
plot_kwargs = {'dpi': 400, 'bbox_inches': 'tight'}
caribbean_epsg = 32620


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
        admin_areas.loc[:, "DIS"] = admin_areas.apply(lambda x: str(num2words(x["DIS"])).title(), axis=1)
        admin_areas.rename(columns={"DIS": "school_district"},inplace=True)
        
    admin_areas = admin_areas.to_crs(epsg=caribbean_epsg)
    admin_areas.loc[:, "school_district"] = admin_areas["school_district"].astype(str).str.replace("Saint","St.")

    # divide road dataframe into intra-admin area roads
    roads_by_district = gpd.sjoin(roads, admin_areas, how="inner", predicate='within').reset_index(drop=True).drop('index_right', axis=1)

    paths_df = []
    road_nets = []
    pathname = os.path.join(outdir, 'infrastructure', 'transport', f'{COUNTRY}_schools_pathdata_{COST}_{THRESH}')

    # process road network and recalculate paths (if RECALCULATE_PATHS)
    for district in [*roads_by_district['school_district'].unique()]:
        # subset geodataframes to district
        roads_district = roads_by_district[roads_by_district['school_district'] == district].copy()
        road_net_dist = road_net.edge_subgraph([(u, v) for u, v in zip(roads_district['from_node'], roads_district['to_node'])])
        schools_district = schools[schools['school_district'] == district].copy()

        # get all node geometries
        node_geoms = {node: Point(line.coords[-1]) for node, line in zip(roads_district['to_node'], roads_district['geometry'])}
        node_geoms = node_geoms | {node: Point(line.coords[0]) for node, line in zip(roads_district['from_node'], roads_district['geometry']) if node not in node_geoms.keys()}
        nodes_gdf = gpd.GeoDataFrame.from_dict(node_geoms, orient='index').reset_index().rename(columns={'index': 'node', 0: 'geometry'}).set_geometry('geometry')
        
        # find nearest nodes to each school building
        schools_district.loc[:, 'nearest_node'] = schools_district.apply(lambda row: get_nearest_values(row, nodes_gdf, 'node'), axis=1)
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
        tfdf.test_pop_assignment(road_net_dist, nearest_nodes, rename_schools, 'assigned_students')
        # test_plot(nodes_gdf, nearest_nodes)

        # get path and flux data
        if RECALCULATE_PATHS:
            path_df = tfdf.get_flux_data(road_net_dist, COST, THRESH, ZETA, origin_class='domestic', dest_class='school', class_str='class')
            path_df.loc[:, 'school_district'] = district
            paths_df.append(path_df)

        road_nets.append(road_net)

    # get path data and subdivided road network
    if RECALCULATE_PATHS:
        pd.concat(paths_df).to_parquet(path=f"{pathname}.parquet", index=True)
    else:
        paths_df = pd.read_parquet(path=f"{pathname}.parquet", engine="fastparquet")
    school_road_net = nx.compose_all(road_nets)

    # add traffic to road edges
    pathname = os.path.join(outdir, 'infrastructure', 'transport', f"{COUNTRY}_schools_traffic_{COST}_{THRESH}")
    traffic_df = tfdf.get_flow_on_edges(paths_df, 'edge_id', 'edge_path', 'flux')
    traffic_df = traffic_df.rename(columns={'flux': 'traffic'})
    roads_traffic = roads_by_district.merge(traffic_df, how='left', on='edge_id')
    tfdf.test_traffic_assignment(roads_traffic, roads_by_district)  # check number of edges unchanged
    roads_traffic.to_file(filename=f"{pathname}.gpkg", driver="GPKG", layer="roads")

    # plot example of school traffic
    fig = tfdf.plot_district_traffic(roads_traffic, schools, 'Two', "school_district", "assigned_students")
    fig.savefig(os.path.join(figdir, f"{COUNTRY}_schools_traffic_zoom_{COST}_{THRESH}.png"), **plot_kwargs)
        
    # set up simulated disruption
    scenario_dict = tfdf.simulate_disruption(roads_traffic, seed=2)
    outfile = os.path.join(outdir, 'infrastructure', 'transport', f"{COUNTRY}_school_failure_aggregates_{COST}_{THRESH}.csv")
    path_df_disrupted = tfdf.model_disruption(scenario_dict, paths_df, road_net, outfile, COST)
    roads_disrupted = tfdf.get_traffic_disruption(paths_df, path_df_disrupted, roads_traffic, scenario_dict, COST)

    # plot changes to traffic
    fig = tfdf.plot_failed_edge_traffic(roads_disrupted, scenario_dict, edge_ix=2, buffer=1000)
    fig.savefig(join(figdir, f"{COUNTRY}_schools_traffic_change_zoom_{scenario_dict['scenario_name']}_{COST}_{THRESH}.png"), **plot_kwargs)


if __name__ == "__main__":
    CONFIG = load_config(join("..", "..", ".."))
    main(CONFIG)