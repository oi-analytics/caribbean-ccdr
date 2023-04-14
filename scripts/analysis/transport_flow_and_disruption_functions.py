"""Modified by Alison"""

import sys
import os
os.environ['USE_PYGEOS'] = '0'
from os.path import join
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from collections import defaultdict
import igraph as ig
from collections import defaultdict
from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm


# load roads network
def get_roads(indir, country: str, edge_attrs: list):
    roads = gpd.read_file(os.path.join(indir, "roads.gpkg"))
    roads = roads[roads['iso_code'] == country]

    # generate time column (minutes)
    roads['time_m'] = 60 * roads['length_m'] / (1000 * roads['max_speed'])

    # create NetworkX network and add populations
    voronois = gpd.read_file(os.path.join(indir, f"{country}_roads_voronoi.gpkg"))
    voronois = voronois.set_index('node_id')

    roads['from_pop'] = roads['from_node'].apply(lambda node: voronois.loc[node]['pop_2020'])
    roads['to_pop'] = roads['to_node'].apply(lambda node: voronois.loc[node]['pop_2020'])

    road_net = nx.convert_matrix.from_pandas_edgelist(roads, source='from_node', target='to_node', edge_attr=edge_attrs)

    # remove missing nodes and print which are missing
    missing = []
    for node in road_net.nodes:
        if node in voronois.index:
            road_net.nodes[node]["population"] = voronois.loc[node]['pop_2020']
        else:
            missing.append(node)
    road_net.remove_nodes_from(missing)
    roads = roads[roads.apply(lambda row: (row['from_node'] not in missing) & (row['to_node'] not in missing), axis=1)]

    # print results
    print(f"Number of nodes: {road_net.number_of_nodes()}")
    print(f"Number of edges: {road_net.number_of_edges()}")

    return roads, road_net


def test_pop_assignment(road_net, buildings, renaming_dict):
    """Test building population assignments to road network nodes worked."""
    renaming_dict_r = {value: key for key, value in renaming_dict.items()}
    for building in buildings.itertuples():
        if building.node_id in road_net.nodes:
            assert road_net.nodes[building.node_id]['population'] == building.assigned_students
        else:
            print(f"Missing {building.node_id}, index: {(building.Index)}, road node: {renaming_dict_r[building.node_id]}")

def test_plot(nodes_gdf, buildings, i=0):
    """Plot school and nearest node to make sure assignments look sensible."""
    fig, ax = plt.subplots()
    building = buildings.iloc[i: i+1, :]
    
    nodes_gdf_clipped = nodes_gdf.clip(building.buffer(100))
    nodes_gdf_clipped.plot(ax=ax)
    nodes_gdf_clipped.set_index('node').loc[building['nearest_node'], :].plot(color='red', ax=ax)
    building.plot(ax=ax)
    ax.set_title(f"{building['node_id'].values[0]}: {building['nearest_node'].values[0]}")
    
def plot_district_traffic(roads, buildings, district, district_label, population_label):
    """Plot the traffic in a district"""
    fig, ax = plt.subplots(figsize=(10, 5))

    roads[roads[district_label] == district].plot('traffic', cmap='Spectral_r', ax=ax, legend=True)
    district_buildings = buildings[buildings[district_label] == district]
    district_buildings.set_geometry(district_buildings.buffer(500)).plot(population_label, cmap='Reds', ax=ax, alpha=0.2, zorder=0)
    district_buildings.plot(population_label, cmap='Reds', edgecolor='black', ax=ax, legend=True)

    ax.set_title(f'Traffic in District {district}');
    return fig
    
def test_traffic_assignment(roads_traffic, roads):
    assert len(roads) == len(roads_traffic), 'no road edges should be deleted after assigning traffic'


def simulate_disruption(roads, num_disrupted=10, scenario_name='dummy_scenario', seed=0):
    """Simulate random disruption on a road network dataframe"""
    failed_edges = roads.sample(num_disrupted, axis=0, random_state=seed)
    edge_fail_tuples = [(o, d) for o, d in zip(failed_edges['from_node'], failed_edges['to_node'])]
    edge_fail_ids = [edge_id for edge_id in failed_edges['edge_id']]
    return {"scenario_name": scenario_name, "edge_fail_tuples": edge_fail_tuples, "edge_fail_ids": edge_fail_ids}

def print_disruption_info(road_net, road_net_disrupted, edge_flow_path_indices, edge_fail_ids):
    """Compare number of connected components between disrupted and not-disrupted road networks."""
    
    print(f"Number of paths affected: {sum([len(paths) for edge, paths in edge_flow_path_indices.items() if edge in edge_fail_ids])}")
    
    print("\nOriginal network")
    connected_subgraphs = [len(c) for c in sorted(nx.connected_components(road_net), key=len, reverse=True)]
    print(f"Number of disconnected subgraphs {len(connected_subgraphs)}")
    print(f"Sizes: {connected_subgraphs[:3]} ... etc.")

    print("\nDisrupted network")
    connected_subgraphs = [len(c) for c in sorted(nx.connected_components(road_net_disrupted), key=len, reverse=True)]
    print(f"Number of disconnected subgraphs {len(connected_subgraphs)}")
    print(f"Sizes: {connected_subgraphs[:3]} ... etc.")
    
    
def model_disruption(scenario_dict, paths_df, road_net, outdir,  COUNTRY, COST, C):
    """Model disruption changes to paths"""
    
    # unpack scenario dict
    scenario_name = scenario_dict["scenario_name"]
    edge_fail_tuples = scenario_dict["edge_fail_tuples"]
    edge_fail_ids = scenario_dict["edge_fail_ids"]
    
    edge_flow_path_indices = get_flow_paths_indexes_of_edges(paths_df, 'edge_path')
    
    road_net_disrupted = road_net.copy()
    road_net_disrupted.remove_edges_from(edge_fail_tuples)
    road_net_df = nx.to_pandas_edgelist(road_net)

    print_disruption_info(road_net, road_net_disrupted, edge_flow_path_indices, edge_fail_ids)
    
    edge_fail_dict = igraph_scenario_edge_failures(road_net_df, edge_fail_ids,
    paths_df, edge_flow_path_indices, 'edge_path', COST, 'flux')

    path_df_disrupted = pd.DataFrame(edge_fail_dict)
    path_df_disrupted[f"rerouting_loss_person_{COST}"] = (1 - path_df_disrupted["no_access"]) * (path_df_disrupted[f"new_{COST}"] - path_df_disrupted[COST]) * path_df_disrupted['flux']
    path_df_disrupted["lost_flux"] = path_df_disrupted["no_access"] * path_df_disrupted['flux']
    path_df_disrupted["rerouted_flux"] = (1 - path_df_disrupted["no_access"]) * path_df_disrupted['flux']
    path_df_disrupted['failed_edges'] = path_df_disrupted['failed_edges'].astype(str)
    aggregated_disruption = path_df_disrupted.groupby(['failed_edges'])[f"rerouting_loss_person_{COST}", "lost_flux", "rerouted_flux"].sum().reset_index()
    aggregated_disruption['scenario_name'] = scenario_name
    # Should probbaly rename csv file below
    outpath = join(outdir, f"{COUNTRY}_failure_aggregates_{COST}_{C}.csv")
    aggregated_disruption.to_csv(outpath, index=False)
    print(f"\nSaved disruption file to {outpath}.")
    return path_df_disrupted


def get_traffic_disruption(path_df_orig, path_df_disrupted, roads_traffic_orig, scenario_dict, COST):
    """Model traffic change from rerouting in disruption scenario."""
    scenario_name = scenario_dict["scenario_name"]
    edge_fail_ids = scenario_dict["edge_fail_ids"]
    
    # merge with old path_df
    path_df_disrupted = path_df_orig.merge(path_df_disrupted, on=['origin_id', 'destination_id', 'flux', COST], how='left')
    path_df_disrupted['new_path'] = path_df_disrupted['new_path'].fillna(path_df_disrupted['edge_path'])
    path_df_disrupted[f'delta_{COST}'] = path_df_disrupted[f'new_{COST}'] - path_df_disrupted[COST]
    
    # recalculate traffic
    traffic_df_disrupted = get_flow_on_edges(path_df_disrupted, 'edge_id', 'new_path', 'flux')
    traffic_df_disrupted = traffic_df_disrupted.rename(columns={'flux': 'traffic'})
    traffic_df_disrupted[f'affected_{scenario_name}'] = 1

    # merge into one road dataframe
    roads_disrupted = roads_traffic_orig.merge(traffic_df_disrupted, how='left', on='edge_id', suffixes=(None, f'_{scenario_name}'))
    roads_disrupted[f'delta_traffic_{scenario_name}'] = roads_disrupted[f'traffic_{scenario_name}'] - roads_disrupted['traffic']
    roads_disrupted[f'affected_{scenario_name}'] = roads_disrupted[f'affected_{scenario_name}'].replace(np.nan, 0.0).astype(bool)
    roads_disrupted['failed'] = roads_disrupted['edge_id'].apply(lambda x: x in edge_fail_ids)
    
    return roads_disrupted


def plot_failed_edge_traffic(roads_disrupted, scenario_dict, COUNTRY, COST, C, edge_ix=0, buffer=1000, cmap_kwargs={'cmap': 'Spectral_r'}):
    scenario_name = scenario_dict['scenario_name']
    failed_edge = scenario_dict['edge_fail_ids'][edge_ix]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    failed_edge = roads_disrupted.set_index('edge_id').loc[[failed_edge]]
    buffer_geom = failed_edge.buffer(buffer).geometry
    roads_disrupted.clip(buffer_geom).plot('traffic', ax=axs[0], **cmap_kwargs, vmin=-100, vmax=100)

    failed_edge.buffer(0.0005).plot(color='red', alpha=0.6, ax=axs[0])
    failed_edge.plot(color='red', ax=axs[0])

    roads_disrupted.clip(buffer_geom).plot(f'traffic_{scenario_name}', ax=axs[1], **cmap_kwargs, vmin=-100, vmax=100, legend=True)

    # plot change
    delta_std = roads_disrupted.clip(buffer_geom)[f'delta_traffic_{scenario_name}'].std()
    roads_disrupted.clip(buffer_geom).plot(f'delta_traffic_{scenario_name}', ax=axs[2], **cmap_kwargs, vmin=-3*delta_std, vmax=3*delta_std, legend=True)

    # set titles
    axs[0].set_title('original traffic')
    axs[1].set_title('disrupted traffic')
    axs[2].set_title(r'$\Delta$ traffic');

    for ax in axs:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
        
    return fig

def get_flux_data(G, cost, C, zeta, population="population", pop_thresh=5, origin_class=None, dest_class=None, class_str=''):
    """
    TODO: replace nx.shortest_paths.single_source_dijkstra with a function that allows a list of targets.
    
    Estimate total traffic flows using radiation model and cost function on edges, based on [1].

    Parameters:
    -----------
    origin_class : str
        asserts all source nodes belong to specific class
    dest_class : str
        asserts all destination nodes belong to specific class
    """

    fluxes = {}
    paths = {}
    costs = {}
    
    b_nodes = [node for node in G.nodes if G.nodes[node].get(class_str) == dest_class]

    for a in (pbar := tqdm(G.nodes, desc='processing paths from source nodes', total=G.number_of_nodes())):
        if G.nodes[a].get(class_str) == origin_class:
            m_a = G.nodes[a][population]
            if m_a > pop_thresh:
                # calculate each node's predecessors and its the cost of its shortest path to the source node
                costs_a, paths_a = nx.shortest_paths.single_source_dijkstra(G, a, cutoff=C, weight=cost)
                for b in paths_a.keys():
                    if b != a and G.nodes[b].get(class_str) == dest_class:
                        n_b = G.nodes[b][population]
                        if n_b > pop_thresh:
                            c_ab = costs_a[b]
                            costs_ab = {key: value for key, value in costs_a.items() if key not in [a, b]}
                            s_ab = sum([G.nodes[node][population] for node, c_v in costs_ab.items() if c_v <= c_ab])
                            phi_ab = zeta * (m_a**2 * n_b) / ((m_a + s_ab) * (m_a + s_ab + n_b))
                            
                            # TODO: assuming only one shortest path each time
                            path_ab = paths_a[b]
                            path_eids = [G[i][j]['edge_id'] for i, j in zip(path_ab[:-1], path_ab[1:])]
                            
                            fluxes[(a, b)] = {'flux': phi_ab}
                            paths[(a, b)] = {'edge_path': path_eids}
                            costs[(a, b)] = {cost: c_ab}
            
    flux_df = pd.DataFrame.from_dict(fluxes, orient='index')
    path_df = pd.DataFrame.from_dict(paths, orient='index')
    cost_df = pd.DataFrame.from_dict(costs, orient='index')
    df = pd.concat([path_df, cost_df, flux_df], axis=1).reset_index()
    df = df.rename(columns={'level_0': 'origin_id', 'level_1': 'destination_id'})
    return df


def get_flow_on_edges(save_paths_df, edge_id_column, edge_path_column, flow_column):
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
        for item in getattr(row, edge_path_column):
            edge_flows[item] += getattr(row, flow_column)

    return pd.DataFrame([(k, v) for k, v in edge_flows.items()], columns=[edge_id_column, flow_column])


def get_flow_paths_indexes_of_edges(flow_dataframe, path_string):
    """Get indexes of paths containing the edge for each edge.

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
    for row in flow_dataframe.itertuples():
        for path in getattr(row, path_string):
            edge_path_index[path].append(row.Index)

    del flow_dataframe
    return edge_path_index


def igraph_scenario_edge_failures(network_df_in, edge_failure_set,
    flow_dataframe, edge_flow_path_indexes, path_criteria,
    cost_criteria, flow_column):
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
    edge_path_index = list(set(list(chain.from_iterable([path_idx for edge, path_idx in edge_flow_path_indexes.items() if edge in edge_failure_set]))))


    if edge_path_index:
        
        select_flows = flow_dataframe[flow_dataframe.index.isin(edge_path_index)]
        del edge_path_index
        network_graph = ig.Graph.TupleList(network_df_in[~network_df_in['edge_id'].isin(edge_failure_set)].itertuples(
            index=False), edge_attrs=list(network_df_in.columns)[2:])

        A = sorted(network_graph.connected_components().subgraphs(),key=lambda l:len(l.es['edge_id']),reverse=True)
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
                    for destination, cost, flow in zip(destinations, costs, flows):
                        paths = network_graph.get_shortest_paths(origin, destination, weights=cost_criteria, output="epath")
                        assert len(paths) == 1, f"{len(paths)} shortest paths found. Code only works for 1."
                        path = paths[0]
                        new_cost = sum(network_graph.es[edge][cost_criteria] for edge in path)
                        new_path = [network_graph.es[edge]['edge_id'] for edge in path]

                        edge_fail_dictionary.append({'failed_edges': edge_failure_set,
                                                     'origin_id': origin, 
                                                    'destination_id': destination,
                                                    "new_path": new_path,
                                                    cost_criteria: cost,
                                                    flow_column: flow,
                                                    f"new_{cost_criteria}": new_cost,
                                                    'no_access': 0})
                    del destinations, paths
                del origins
                po_access = po_access.reset_index()
                po_access['access'] = 1
                access_flows.append(po_access[['origin_id','destination_id','access']])
            del po_access

        del A

        if len(access_flows):
            access_flows = pd.concat(access_flows, axis=0, sort=False, ignore_index=True)
            select_flows = pd.merge(select_flows, access_flows, how='left', on=['origin_id', 'destination_id']).fillna(0)
        else:
            select_flows['access'] = 0

        # process OD-pairs where there is now no route between them
        no_access = select_flows[select_flows['access'] == 0]
        if len(no_access.index) > 0:
            for value in no_access.itertuples():
                edge_fail_dictionary.append({'failed_edges': edge_failure_set,
                                             'origin_id': getattr(value,'origin_id'),
                                            'destination_id': getattr(value,'destination_id'),
                                            cost_criteria: getattr(value, cost_criteria),
                                            flow_column: getattr(value, flow_column),
                                            "new_path": [],
                                            f"new_{cost_criteria}": 0,
                                            'no_access': 1})

        del no_access, select_flows

    return edge_fail_dictionary