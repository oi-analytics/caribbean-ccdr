"""Failure analysis of transport networks

"""
import ast
import copy
import csv
import itertools
import math
import operator
import os
import sys

import igraph as ig
import numpy as np
import pandas as pd
from analysis_utils import load_config
from transport_flow_and_disruption_functions import *



def main(config):
    """Estimate failures
    """
    data_path, output_path = config['paths']['data'], config['paths']['output']

    fail_output_path = os.path.join(output_path, 'failure_results')
    if os.path.exists(fail_output_path) == False:
        os.mkdir(fail_output_path)

    network_data_path = os.path.join(data_path,"infrastructure/transport")
    # Load network dataframe
    print ('* Loading network GeoDataFrame')
    gdf_edges = gpd.read_file(os.path.join(network_data_path,'roads.gpkg'),layer="edges")


    # Load flow paths
    print ('* Loading flow paths')
    flow_df = pd.read_parquet(os.path.join(flow_paths_data))

    # Read failure scenarios
    print ('* Reading failure scenarios')
    fail_df = pd.read_csv(os.path.join('path/to/failed/edges'))
    print ('Number of failure scenarios',len(fail_df))

    path_types = "edge_path"
    cost_types = "time"
    flow_types = "traffic"

    edge_path_idx = get_flow_paths_indexes_of_edges(flow_df,path_types)
    print ('* Performing failure analysis')
    ef_list = []
    for f_edge in range(len(ef_sc_list)):
        fail_edge = ef_sc_list[f_edge]
        if isinstance(fail_edge,list) == False:
            fail_edge = [fail_edge]
        
        ef_dict = igraph_scenario_edge_failures(
                gdf_edges, fail_edge, flow_df,edge_path_idx,
                path_types, 
                cost_types,flow_types)

        if ef_dict:
            ef_list += ef_dict

        print(f"Done with scenario {f_edge} out of {len(ef_sc_list)}")
    df = pd.DataFrame(ef_list)
    print ('* Assembling failure results')
    df["rerouting_loss_person_hr"] = (1 - df["no_access"])*(df[f"new_{cost_types}"] - df[{cost_types}])*df[flow_types]
    df["traffic_loss_person"] = df["no_access"]*df[flow_types]
    df = df.groupby(["fail_edges"])["rerouting_loss_person_hr","traffic_loss_person"].sum().reset_index()
    # Should probbaly rename csv file below
    df.to_csv(os.path.join(fail_output_path,"road_failure.csv"),index=False)



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
