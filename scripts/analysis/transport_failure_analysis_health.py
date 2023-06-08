"""Failure analysis of transport networks

Timings:
--------
Truncating paths with threshold 99%.
Truncating paths with threshold 99%.
Number of paths before: 4,408.
Number of paths after: 75.
Took 6.1864 seconds. --- decided not to truncate as quick enough anyway.


Took 77.1721 seconds (1.3 minutes) without truncation.

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

# set country using command likne
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--country', type=str, help='what country to run script on', default='LCA')
args = parser.parse_args()
COUNTRY= args.country
print(f'Processing health data for country: {COUNTRY}')

# global settings
COST = 'time_m'
THRESH = 60
TRUNC_THRESH = 1
ZETA = 1
RECALCULATE_PATHS = True
RECALCULATE_TRAFFIC = False
EDGE_ATTRS = ['edge_id', 'length_m', 'time_m']
plot_kwargs = {'dpi': 400, 'bbox_inches': 'tight'}
caribbean_epsg = 32620


def main(CONFIG):
    # set up directories
    datadir, resultsdir, figdir = CONFIG['paths']['data'], CONFIG['paths']['results'], CONFIG['paths']['figures']

    # load roads
    roads, road_net = tfdf.get_roads(os.path.join(datadir, 'infrastructure', 'transport'), COUNTRY, EDGE_ATTRS)
    roads = roads.to_crs(caribbean_epsg)

    # load health data
    health = gpd.read_file(os.path.join(datadir, 'infrastructure', 'social', f'{COUNTRY.lower()}_health.gpkg'))
    health = health[health['iso_code'] == COUNTRY].copy()
    health = health.to_crs(caribbean_epsg)

    # step 1: get path and flux dataframe and save
    path_df, road_net = tfdf.process_health_fluxes(roads, road_net, health, resultsdir, COUNTRY, COST, THRESH, ZETA, RECALCULATE_PATHS, TRUNC_THRESH)
    import pdb; pdb.set_trace()

    # step 2: model disruption
    outfile = os.path.join(resultsdir, "transport", "disruption results", f"{COUNTRY.lower()}_health_roads_edges_sector_damages_with_rerouting")
    disruption_file = os.path.join(resultsdir, "transport", "disruption results", f"{COUNTRY.lower()}_roads_edges_sector_damages.parquet")
    disruption_df = pd.read_parquet(disruption_file)
    disruption_df = tfdf.get_disruption_stats(disruption_df, path_df, road_net, COST)
    disruption_df.to_parquet(f"{outfile}.parquet")

    # step 3: add traffic to road edges
    # TODO: outdated below here
    if RECALCULATE_TRAFFIC:
        pathname = os.path.join(resultsdir, 'transport', 'traffic', f"{COUNTRY}_health_traffic_{COST}_{THRESH}")
        traffic_df = tfdf.get_flow_on_edges(path_df, 'edge_id', 'edge_path', 'flux')
        traffic_df = traffic_df.rename(columns={'flux': 'traffic'})
        roads_traffic = roads.merge(traffic_df, how='left', on='edge_id')
        tfdf.test_traffic_assignment(roads_traffic, roads)  # check number of edges unchanged
        roads_traffic.to_file(filename=f"{pathname}.gpkg", driver="GPKG", layer="roads")


if __name__ == '__main__':
    CONFIG = load_config(os.path.join("..", "..", ".."))
    main(CONFIG)
