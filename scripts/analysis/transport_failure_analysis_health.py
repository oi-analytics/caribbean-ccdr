"""Failure analysis of transport networks

"""
import os
os.environ['USE_PYGEOS'] = '0'

import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from geospatial_utils import load_config
from analysis_utils import get_nearest_values
import transport_flow_and_disruption_functions as tfdf


# global settings
COUNTRY = 'LCA'
COST = 'time_m'
THRESH = 30
ZETA = 1
RECALCULATE_PATHS = True
EDGE_ATTRS = ['edge_id', 'length_m', 'time_m']
plot_kwargs = {'dpi': 400, 'bbox_inches': 'tight'}
caribbean_epsg = 32620

# health stats
LCA_stats = {'beds_per_1000': 1.3,
             'ref_year': 2017,
             'pop_ref_year': 177163,
             'source': 'https://data.worldbank.org/indicator/SH.MED.BEDS.ZS?locations=LC'}
LCA_stats['total_beds'] = LCA_stats['beds_per_1000'] * (LCA_stats['pop_ref_year'] / 1000 )


def main(CONFIG):
    # set up directories
    indir, outdir, figdir = CONFIG['paths']['incoming_data'], CONFIG['paths']['data'], CONFIG['paths']['figures']

    # load roads
    roads, road_net = tfdf.get_roads(os.path.join(outdir, 'infrastructure', 'transport'), COUNTRY, EDGE_ATTRS)

    # load health data
    health = gpd.read_file(os.path.join(outdir, 'infrastructure', 'social', 'health.gpkg'))
    health = health[health['iso_code'] == COUNTRY].copy()
    health = health.to_crs(caribbean_epsg)
    health = health.rename(columns={'capacity:persons': 'capacity'})

    # assign capacities
    hospital_ix = health[health['amenity'] == 'hospital'].index
    other_ix = health[health['amenity'] != 'hospital'].index
    nhospitals = len(hospital_ix)
    nother = len(other_ix)
    total = (2 * nhospitals) + nother
    health.loc[hospital_ix, 'capacity'] = 2 * LCA_stats['total_beds'] / total
    health.loc[other_ix, 'capacity'] = LCA_stats['total_beds'] / total

    # get all node geometries
    node_geoms = {node: Point(line.coords[-1]) for node, line in zip(roads['to_node'], roads['geometry'])}
    node_geoms = node_geoms | {node: Point(line.coords[0]) for node, line in zip(roads['from_node'], roads['geometry']) if node not in node_geoms.keys()}
    nodes_gdf = gpd.GeoDataFrame.from_dict(node_geoms, orient='index').reset_index().rename(columns={'index': 'node', 0: 'geometry'}).set_geometry('geometry')

    # find nearest nodes to each school building
    health['nearest_node'] = health.apply(lambda row: get_nearest_values(row, nodes_gdf, 'node'), axis=1)
    aggfunc = {'node_id': lambda x : '_and_'.join(x), 'capacity': sum}
    nearest_nodes = health[['nearest_node', 'node_id', 'capacity']].groupby('nearest_node').agg(aggfunc).reset_index()
    # tfdf.test_plot(nodes_gdf, health, **{'aspect': 1})

    # replace nodes with health in road network
    rename_health = {node_id: hospital_id for node_id, hospital_id in zip(nearest_nodes['nearest_node'], nearest_nodes['node_id'])}
    hospital_pops = {hospital_id: hospital_pop for hospital_id, hospital_pop in zip(nearest_nodes['node_id'], nearest_nodes['capacity'])}
    hospital_classes = {hospital_id: "hospital" for hospital_id in nearest_nodes['node_id']}
    road_net = nx.relabel_nodes(road_net, rename_health, copy=True)
    nx.set_node_attributes(road_net, hospital_pops, name="population")
    nx.set_node_attributes(road_net, "domestic", name="class")
    nx.set_node_attributes(road_net, hospital_classes, name="class")
    tfdf.test_pop_assignment(road_net, nearest_nodes, rename_health, 'capacity')

    # get path and flux dataframe and save
    pathname = os.path.join(outdir, 'infrastructure', 'transport', f'{COUNTRY}_health_pathdata_{COST}_{THRESH}')
    if RECALCULATE_PATHS:
        paths_df = tfdf.get_flux_data(road_net, COST, THRESH, ZETA, origin_class='domestic', dest_class='hospital', class_str='class')
        paths_df.to_parquet(path=f"{pathname}.parquet", index=True)
        print(f"Paths data saved as {pathname}.parquet.")
    else:
        paths_df = pd.read_parquet(path=f"{pathname}.parquet", engine="fastparquet")

    # add traffic to road edges
    pathname = os.path.join(outdir, 'infrastructure', 'transport', f"{COUNTRY}_health_traffic_{COST}_{THRESH}")
    traffic_df = tfdf.get_flow_on_edges(paths_df, 'edge_id', 'edge_path', 'flux')
    traffic_df = traffic_df.rename(columns={'flux': 'traffic'})
    roads_traffic = roads.merge(traffic_df, how='left', on='edge_id')
    tfdf.test_traffic_assignment(roads_traffic, roads)  # check number of edges unchanged
    roads_traffic.to_file(filename=f"{pathname}.gpkg", driver="GPKG", layer="roads")
    
    # set up a simulated disruption
    scenario_dict = tfdf.simulate_disruption(roads_traffic, seed=2)
    outfile = os.path.join(outdir, 'infrastructure', 'transport', f"{COUNTRY}_health_failure_aggregates_{COST}_{THRESH}.csv")
    path_df_disrupted = tfdf.model_disruption(scenario_dict, paths_df, road_net, outfile, COST)
    if path_df_disrupted is not None:
        roads_disrupted = tfdf.get_traffic_disruption(paths_df, path_df_disrupted, roads_traffic, scenario_dict, COST)

        # plot some disrupted edges
        fig = tfdf.plot_failed_edge_traffic(roads_disrupted, scenario_dict, edge_ix=2, buffer=100)
        fig.savefig(os.path.join(figdir, f"{COUNTRY}_health_traffic_change_zoom_{scenario_dict['scenario_name']}_{COST}_{THRESH}.png"), **plot_kwargs)

if __name__ == '__main__':
    CONFIG = load_config(os.path.join("..", "..", ".."))
    main(CONFIG)
