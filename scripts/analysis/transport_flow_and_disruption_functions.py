import sys
import os
import json

import pandas as pd
import geopandas as gpd
from collections import defaultdict
import igraph as ig
from collections import defaultdict
from itertools import chain

def get_flow_on_edges(save_paths_df,edge_id_column,edge_path_column,
    flow_column):
    """Get flows from paths onto edges

    Parameters
    ---------
    save_paths_df
        Pandas DataFrame of OD flow paths and their flow values
    edge_id_column
        String name of ID column of edge dataset
    edge_path_column
        String name of column which contains edge paths
    flow_column
        String name of column which contains flow values

    Result
    -------
    DataFrame with edge_ids and their total flows
    """
    """Example:
        save_path_df:
            origin_id | destination_id | edge_path          |  flux (or traffic)
            node_1      node_2          ['edge_1','edge_2']     10    
            node_1      node_3          ['edge_1','edge_3']     20

        edge_id_column = "edge_id"
        edge_path_column = "edge_path"
        flow_column = "traffic"

        Result
            edge_id | flux (or traffic)
            edge_1      30
            edge_2      10
            edge_3      20
    """
    edge_flows = defaultdict(float)
    for row in save_paths_df.itertuples():
        for item in getattr(row,edge_path_column):
            edge_flows[item] += getattr(row,flow_column)

    return pd.DataFrame([(k,v) for k,v in edge_flows.items()],columns=[edge_id_column,flow_column])

def get_flow_paths_indexes_of_edges(flow_dataframe,path_criteria):
    """Get indexes of paths for each

    Parameters
    ---------
    flow_dataframe
        Pandas DataFrame of OD flow paths and their flow values
    path_criteria
        String name of column which contains edge paths

    Result
    -------
    Dictionary with edge_ids as keys and the list of path indexes with these edge_ids
    """
    """Example:
        flow_dataframe:
            origin_id | destination_id | edge_path          |  flux (or traffic)
            node_1      node_2          ['edge_1','edge_2']     10    
            node_1      node_3          ['edge_1','edge_3']     20

        path_criteria = "edge_path"

        Result
            {'edge_1':[0,1],'edge_2':[0],'edge_3':[1]}
    """
    edge_path_index = defaultdict(list)
    for v in flow_dataframe.itertuples():
        for k in getattr(v,path_criteria):
            edge_path_index[k].append(v.Index)

    del flow_dataframe
    return edge_path_index

def igraph_scenario_edge_failures(network_df_in, edge_failure_set,
    flow_dataframe,edge_flow_path_indexes,path_criteria,
    cost_criteria,flow_column):
    """Estimate network impacts of each failures
    When the tariff costs of each path are fixed by vehicle weight

    Parameters
    ---------
    network_df_in - Pandas DataFrame of network
    edge_failure_set - List of string edge ID's which has failed
    flow_dataframe - Pandas DataFrame of list of edge paths
    path_criteria - String name of column of edge paths in flow dataframe
    tons_criteria - String name of column of path tons in flow dataframe
    cost_criteria - String name of column of path costs in flow dataframe
    time_criteria - String name of column of path travel time in flow dataframe


    Returns
    -------
    edge_failure_dictionary : list[dict]
        With attributes
        edge_id - String name or list of failed edges
        origin - String node ID of Origin of disrupted OD flow
        destination - String node ID of Destination of disrupted OD flow
        no_access - Boolean 1 (no reroutng) or 0 (rerouting)
        new_cost - Float value of estimated cost of OD journey after disruption
        new_path - List of string edge ID's of estimated new route of OD journey after disruption
    """
    edge_fail_dictionary = []
    edge_path_index = list(set(list(chain.from_iterable([path_idx for path_key,path_idx in edge_flow_path_indexes.items() if path_key in edge_failure_set]))))

    if edge_path_index:
        select_flows = flow_dataframe[flow_dataframe.index.isin(edge_path_index)]
        del edge_path_index
        network_graph = ig.Graph.TupleList(network_df_in[~network_df_in['edge_id'].isin(edge_failure_set)].itertuples(
            index=False), edge_attrs=list(network_df_in.columns)[2:])

        first_edge_id = edge_failure_set
        del edge_failure_set
        A = sorted(network_graph.clusters().subgraphs(),key=lambda l:len(l.es['edge_id']),reverse=True)
        access_flows = []
        edge_fail_dictionary = []
        for i in range(len(A)):
            network_graph = A[i]
            nodes_name = np.asarray([x['name'] for x in network_graph.vs])
            po_access = select_flows[(select_flows['origin_id'].isin(nodes_name)) & (
                    select_flows['destination_id'].isin(nodes_name))]

            if len(po_access.index) > 0:
                po_access = po_access.set_index('origin_id')
                origins = list(set(po_access.index.values.tolist()))
                for o in range(len(origins)):
                    origin = origins[o]
                    destinations = po_access.loc[[origin], 'destination_id'].values.tolist()
                    costs = po_access.loc[[origin], cost_criteria].values.tolist()
                    flows = po_access.loc[[origin], flow_column].values.tolist()
                    paths = network_graph.get_shortest_paths(
                        origin, destinations, weights=cost_criteria, output="epath")
                    for p in range(len(paths)):
                        new_gcost = 0
                        new_path = []
                        for n in paths[p]:
                            new_gcost += network_graph.es[n][cost_criteria]
                            new_path.append(network_graph.es[n]['edge_id'])
                        edge_fail_dictionary.append({'failed_edges': first_edge_id, 
                                                    'origin_id': origin, 
                                                    'destination_id': destinations[p],
                                                    cost_criteria:costs[p],
                                                    f"new_{cost_criteria}":new_gcost,
                                                    flow_column:flows[p],
                                                    'no_access': 0})
                    del destinations, tons, paths
                del origins
                po_access = po_access.reset_index()
                po_access['access'] = 1
                access_flows.append(po_access[['origin_id','destination_id','access']])
            del po_access

        del A

        if len(access_flows):
            access_flows = pd.concat(access_flows,axis=0,sort='False', ignore_index=True)
            select_flows = pd.merge(select_flows,access_flows,how='left',on=['origin_id','destination_id']).fillna(0)
        else:
            select_flows['access'] = 0

        no_access = select_flows[select_flows['access'] == 0]
        if len(no_access.index) > 0:
            for value in no_access.itertuples():
                edge_fail_dictionary.append({'failed_edges': first_edge_id, 
                                            'origin_id': getattr(value,'origin_id'),
                                            'destination_id': getattr(value,'destination_id'),
                                            cost_criteria:getattr(value,cost_criteria),
                                            f"new_{cost_criteria}": 0,
                                            flow_column:getattr(value,flow_column),
                                            'no_access': 1})

        del no_access, select_flows

    return edge_fail_dictionary