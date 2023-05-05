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
THRESH = 60
ZETA = 1
RECALCULATE_PATHS = False
RECALCULATE_TRAFFIC = False
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
    indir, datadir, figdir = CONFIG['paths']['incoming_data'], CONFIG['paths']['data'], CONFIG['paths']['figures']

    # load roads
    roads, road_net = tfdf.get_roads(os.path.join(datadir, 'infrastructure', 'transport'), COUNTRY, EDGE_ATTRS)

    # load health data
    health = gpd.read_file(os.path.join(datadir, 'infrastructure', 'social', 'health.gpkg'))
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

    # step 1: get path and flux dataframe and save
    path_df, road_net = tfdf.process_health_fluxes(roads, road_net, health, datadir, COUNTRY, COST, THRESH, ZETA, RECALCULATE_PATHS)

    # step 2: model disruption
    outfile = os.path.join(datadir, "infrastructure", "transport", f"{COUNTRY.lower()}_health_roads_edges_sector_damages_with_roads")
    disruption_file = os.path.join(datadir, "infrastructure", "transport", f"{COUNTRY.lower()}_roads_edges_sector_damages.parquet")
    disruption_df = pd.read_parquet(disruption_file)
    disruption_df = tfdf.get_disruption_stats(disruption_df, path_df, road_net, COST)
    disruption_df.head(10).to_csv(f"{outfile}.csv")
    disruption_df.to_parquet(f"{outfile}.parquet")

    # step 3: add traffic to road edges
    if RECALCULATE_TRAFFIC:
        pathname = os.path.join(datadir, 'infrastructure', 'transport', f"{COUNTRY}_health_traffic_{COST}_{THRESH}")
        traffic_df = tfdf.get_flow_on_edges(path_df, 'edge_id', 'edge_path', 'flux')
        traffic_df = traffic_df.rename(columns={'flux': 'traffic'})
        roads_traffic = roads.merge(traffic_df, how='left', on='edge_id')
        tfdf.test_traffic_assignment(roads_traffic, roads)  # check number of edges unchanged
        roads_traffic.to_file(filename=f"{pathname}.gpkg", driver="GPKG", layer="roads")
    
    # # set up a simulated disruption
    # scenario_dict = tfdf.simulate_disruption(roads_traffic, seed=2)
    # outfile = os.path.join(datadir, 'infrastructure', 'transport', f"{COUNTRY}_health_failure_aggregates_{COST}_{THRESH}.csv")
    # path_df_disrupted = tfdf.model_disruption(scenario_dict, paths_df, road_net, outfile, COST)
    # if path_df_disrupted is not None:
    #     roads_disrupted = tfdf.get_traffic_disruption(paths_df, path_df_disrupted, roads_traffic, scenario_dict, COST)

    #     # plot some disrupted edges
    #     fig = tfdf.plot_failed_edge_traffic(roads_disrupted, scenario_dict, edge_ix=2, buffer=100)
    #     fig.savefig(os.path.join(figdir, f"{COUNTRY}_health_traffic_change_zoom_{scenario_dict['scenario_name']}_{COST}_{THRESH}.png"), **plot_kwargs)

if __name__ == '__main__':
    CONFIG = load_config(os.path.join("..", "..", ".."))
    main(CONFIG)
