"""
Testing code for transport failure analysis.

Define dummy network with population 10 at nodes 1 and 3.
Edge path 1: 1 -> 2 -> 3 with each edge costing 1, ['e1', 'e2']
Edge path 2: 1 -> 4 -> 5 -> 3 with each edge costing 2, ['e3', 'e4'. 'e5']
Test removing 'e1'

>> python -m pytest -x    

"""

try:
    import pytest
    import os
    os.environ['USE_PYGEOS'] = '0'
    import numpy as np
    import networkx as nx
    import pandas as pd
    from tqdm import tqdm
    from geospatial_utils import load_config
    import transport_flow_and_disruption_functions as tfdf

    CONFIG = load_config(os.path.join("..", "..", ".."))
    COUNTRY = 'test'
    COST = 'time_m'
    THRESH = 30
    ZETA = 1
    EDGE_ATTRS = ['edge_id', 'time_m']
    plot_kwargs = {'dpi': 400, 'bbox_inches': 'tight'}
    caribbean_epsg = 32620
    _, datadir, figdir = CONFIG['paths']['incoming_data'], CONFIG['paths']['data'], CONFIG['paths']['figures']
except:
    pass


def get_dummy_roads():
    """Return a simplistic road network for testing."""
    dummy_roads = pd.DataFrame(columns=['from_node', 'to_node', 'from_pop', 'to_pop', 'time_m'])
    dummy_roads['from_node'] = ['1', '1', '2', '4', '5']
    dummy_roads['to_node'] =  ['2', '4', '3', '5', '3']
    dummy_roads['from_pop'] = [10, 10, 0, 0, 0]
    dummy_roads['to_pop'] = [0, 0, 10, 0, 10]
    dummy_roads['time_m'] = [1, 2, 1, 2, 2]
    dummy_roads['edge_id'] = ['e1', 'e2', 'e3', 'e4', 'e5']
    dummy_road_net = nx.convert_matrix.from_pandas_edgelist(dummy_roads, source='from_node', target='to_node', edge_attr=['edge_id', 'time_m'])
    dummy_road_net.nodes['1']['population'] = 10
    dummy_road_net.nodes['2']['population'] = 0
    dummy_road_net.nodes['3']['population'] = 10
    dummy_road_net.nodes['4']['population'] = 0
    dummy_road_net.nodes['5']['population'] = 0
    # dummy_road_net.nodes['population'] = [10, 0, 10, 0, 0]
    return dummy_roads, dummy_road_net


def get_dummy_disruption():
    """Works with dummy roads."""
    dummy_disruption_df = pd.DataFrame(columns=['asset_set_amin', 'asset_set_mean',	'asset_set_amax'])
    dummy_disruption_df['asset_set_amin'] = [['e1']]
    dummy_disruption_df['asset_set_mean'] = [['e1']]
    dummy_disruption_df['asset_set_amax'] = [['e1']]
    return dummy_disruption_df


def test_import(): 
    import os
    os.environ['USE_PYGEOS'] = '0'
    import numpy as np
    import networkx as nx
    import pandas as pd
    from tqdm import tqdm
    from geospatial_utils import load_config
    import transport_flow_and_disruption_functions as tfdf


def test_env_vars():
    CONFIG = load_config(os.path.join("..", "..", ".."))
    # global settings
    COUNTRY = 'test'
    COST = 'time_m'
    THRESH = 30
    ZETA = 1
    EDGE_ATTRS = ['edge_id', 'time_m']
    plot_kwargs = {'dpi': 400, 'bbox_inches': 'tight'}
    caribbean_epsg = 32620
    _, datadir, figdir = CONFIG['paths']['incoming_data'], CONFIG['paths']['data'], CONFIG['paths']['figures']


def test_assign_fluxes():
    _, road_net = get_dummy_roads()
    path_df = tfdf.get_flux_data(road_net, COST, THRESH, ZETA, other_costs=[])
    assert len(path_df) == 2
    assert path_df.loc[0, 'flux'] == 5
    assert path_df.loc[1, 'flux'] == 5
    assert path_df.loc[0, 'edge_path'] == ['e1', 'e3']
    assert path_df.loc[1, 'edge_path'] == ['e3', 'e1']


def test_traffic_assignments():
    roads, road_net = get_dummy_roads()
    path_df = tfdf.get_flux_data(road_net, COST, THRESH, ZETA, other_costs=[])

    traffic_df = tfdf.get_flow_on_edges(path_df, 'edge_id', 'edge_path', 'flux')
    traffic_df = traffic_df.rename(columns={'flux': 'traffic'})
    roads_traffic = roads.merge(traffic_df, how='left', on='edge_id').replace(np.nan, 0.0)
    tfdf.test_traffic_assignment(roads_traffic, roads)  # check number of edges unchanged

    test_roads_traffic = roads_traffic.set_index(['from_node', 'to_node'])
    assert test_roads_traffic.loc[(1, 2)]['traffic'] == 10.0
    assert test_roads_traffic.loc[(2, 3)]['traffic'] == 10.0
    assert test_roads_traffic.loc[(1, 4)]['traffic'] == 0.0
    assert test_roads_traffic.loc[(4, 5)]['traffic'] == 0.0
    assert test_roads_traffic.loc[(5, 3)]['traffic'] == 0.0


def test_disruption_calculations():
    roads, road_net = get_dummy_roads()
    path_df = tfdf.get_flux_data(road_net, COST, THRESH, ZETA, other_costs=[])
    traffic_df = tfdf.get_flow_on_edges(path_df, 'edge_id', 'edge_path', 'flux')
    traffic_df = traffic_df.rename(columns={'flux': 'traffic'})
    roads_traffic = roads.merge(traffic_df, how='left', on='edge_id').replace(np.nan, 0.0)

    disruption_df = get_dummy_disruption()
    disruption_df = tfdf.get_disruption_stats(disruption_df, path_df, road_net, COST)

    assert disruption_df['trips_lost_amax'][0] == 0.0
    assert disruption_df['%trips_lost_amax'][0] == 0.0
    assert disruption_df['time_m_delta_amax'][0] == 8.0
    assert disruption_df['%time_m_delta_amax'][0] == 2.0
    assert disruption_df['rerouting_loss_person_time_m_amax'][0] == 40.0


if __name__ =="__main__":
    test_disruption_calculations()