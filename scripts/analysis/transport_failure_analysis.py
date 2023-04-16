"""Failure analysis of transport networks

"""
import os
os.environ['USE_PYGEOS'] = '0'

import pandas as pd
from geospatial_utils import load_config
import transport_flow_and_disruption_functions as tfdf


# global settings
COUNTRY = 'LCA'
COST = 'time_m'
THRESH = 5
ZETA = 1
RECALCULATE_PATHS = False
EDGE_ATTRS = ['edge_id', 'length_m', 'time_m']
plot_kwargs = {'dpi': 400, 'bbox_inches': 'tight'}
caribbean_epsg = 32620


def main(CONFIG):
    # set up directories
    indir, outdir, figdir = CONFIG['paths']['incoming_data'], CONFIG['paths']['data'], CONFIG['paths']['figures']

    roads, road_net = tfdf.get_roads(os.path.join(outdir, 'infrastructure', 'transport'), COUNTRY, EDGE_ATTRS)

    # get path and flux dataframe and save
    pathname = os.path.join(outdir, 'infrastructure', 'transport', f'{COUNTRY}_pathdata_{COST}_{THRESH}')
    if RECALCULATE_PATHS:
        path_df = tfdf.get_flux_data(road_net, COST, THRESH, ZETA)
        path_df.to_parquet(path=f"{pathname}.parquet", index=True)
        print(f"Paths data saved as {pathname}.parquet.")
    else:
        path_df = pd.read_parquet(path=f"{pathname}.parquet", engine="fastparquet")

    # add traffic to road edges
    pathname = os.path.join(outdir, 'infrastructure', 'transport', f"{COUNTRY}_traffic_{COST}_{THRESH}")
    traffic_df = tfdf.get_flow_on_edges(path_df, 'edge_id', 'edge_path', 'flux')
    traffic_df = traffic_df.rename(columns={'flux': 'traffic'})
    roads_traffic = roads.merge(traffic_df, how='left', on='edge_id')
    tfdf.test_traffic_assignment(roads_traffic, roads)  # check number of edges unchanged
    roads_traffic.to_file(filename=f"{pathname}.gpkg", driver="GPKG", layer="roads")

    # set up a simulated disruption
    scenario_dict = tfdf.simulate_disruption(roads_traffic, seed=2)
    outfile = os.path.join(outdir, 'infrastructure', 'transport', f"{COUNTRY}_failure_aggregates_{COST}_{THRESH}.csv")
    path_df_disrupted = tfdf.model_disruption(scenario_dict, path_df, road_net, outfile, COST)
    roads_disrupted = tfdf.get_traffic_disruption(path_df, path_df_disrupted, roads_traffic, scenario_dict, COST)

    # plot some disrupted edges
    fig = tfdf.plot_failed_edge_traffic(roads_disrupted, scenario_dict, edge_ix=2, buffer=100)
    fig.savefig(os.path.join(figdir, f"{COUNTRY}_traffic_change_zoom_{scenario_dict['scenario_name']}_{COST}_{THRESH}.png"), **plot_kwargs)

if __name__ == '__main__':
    CONFIG = load_config(os.path.join("..", "..", ".."))
    main(CONFIG)
