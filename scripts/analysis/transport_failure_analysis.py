"""Failure analysis of transport networks

This is taking a very long time and freezing at 0% for the processing step. Loading the pathdata is not the issue. It's hanging on tfdf.model_disruption

Timings:
--------
Truncating paths with threshold 95%.
Number of paths before: 18,166,568.
Number of paths after: 101,661.
Took 90.6058 seconds.
"""
import os
os.environ['USE_PYGEOS'] = '0'

import pandas as pd
from tqdm import tqdm
from geospatial_utils import load_config
import transport_flow_and_disruption_functions as tfdf


# global settings
COUNTRY = 'GRD'
COST = 'time_m'
THRESH = 30
TRUNC_THRESH = .95
ZETA = 1
RECALCULATE_PATHS = True
RECALCULATE_TRAFFIC = False
EDGE_ATTRS = ['edge_id', 'length_m', 'time_m']
plot_kwargs = {'dpi': 400, 'bbox_inches': 'tight'}
caribbean_epsg = 32620


def main(CONFIG):
    # set up directories
    datadir, resultsdir, figdir = CONFIG['paths']['data'], CONFIG['paths']['results'], CONFIG['paths']['figures']
    roads, road_net = tfdf.get_roads(os.path.join(datadir, 'infrastructure', 'transport'), COUNTRY, EDGE_ATTRS)

    # step 1: get path and flux dataframe and save
    pathname = os.path.join(resultsdir, 'transport', 'path and flux data', f'{COUNTRY}_pathdata_{COST}_{THRESH}')
    if RECALCULATE_PATHS:
        path_df = tfdf.get_flux_data(road_net, COST, THRESH, ZETA, other_costs=['length_m'])
        path_df.to_parquet(path=f"{pathname}.parquet", index=True)
        print(f"Paths data saved as {pathname}.parquet.")
    else:
        path_df = pd.read_parquet(path=f"{pathname}.parquet", engine="fastparquet")

    # step 1(b): truncate disruption to remove smallest fluxes
    path_df, fluxes_sorted, flux_percentiles, keep, cutoff = tfdf.truncate_by_threshold(path_df, threshold=TRUNC_THRESH)
    fig = tfdf.plot_path_truncation(fluxes_sorted, flux_percentiles, cutoff, TRUNC_THRESH)
    fig.savefig(os.path.join(figdir, "thresholding.png"), dpi=300, bbox_inches='tight')

    # step 2: model disruption
    outdir = os.path.join(resultsdir, "transport", "disruption results")
    assert os.path.exists(outdir), f"Check results directory path matches {outdir} before running disruption."
    outfile = os.path.join(outdir, f"{COUNTRY.lower()}_roads_edges_sector_damages_with_rerouting")
    disruption_file = os.path.join(outdir, f"{COUNTRY.lower()}_roads_edges_sector_damages.parquet")
    disruption_df = pd.read_parquet(disruption_file)
    disruption_df = tfdf.get_disruption_stats(disruption_df, path_df, road_net, COST)
    disruption_df.to_parquet(f"{outfile}.parquet")

    # step 3: add traffic to road edges
    # TODO: outdated below here
    if RECALCULATE_TRAFFIC:
        pathname = os.path.join(resultsdir, 'transport','traffic', f"{COUNTRY}_traffic_{COST}_{THRESH}")
        traffic_df = tfdf.get_flow_on_edges(path_df, 'edge_id', 'edge_path', 'flux')
        traffic_df = traffic_df.rename(columns={'flux': 'traffic'})
        roads_traffic = roads.merge(traffic_df, how='left', on='edge_id').replace(np.nan, 0)
        tfdf.test_traffic_assignment(roads_traffic, roads)  # check number of edges unchanged
        roads_traffic.to_file(filename=f"{pathname}.gpkg", driver="GPKG", layer="roads")


if __name__ == '__main__':
    CONFIG = load_config(os.path.join("..", "..", ".."))
    main(CONFIG)
