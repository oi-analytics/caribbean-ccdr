import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def get_traffic(G, cost, C, zeta, population="population", verbose=False):
    """
    Estimate total traffic flows using radiation model and cost function on edges, based on [1].
    
    Parameters:
    -----------
    G : nx.Graph
    cost : string
        Travel cost measure, e.g. length (km), time (minutes). String must be an edge attribute of G.
    C : float
        Range limit on how far to build MPTs from each source node.
    zeta : float
        Fraction of travellers in a location.
    population : string
        Label of node weight to use when calculating fluxes.
        
    Examples:
    ---------
    >>> import networkx as nx
    >>> import random
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # make a random graph
    >>> G = nx.random_geometric_graph(20, 0.4)
    >>> for node in G.nodes:
    >>>     G.nodes[node]["population"] = random.randint(10, 1000)
    >>> G, fluxes = get_traffic(G, None, 1000, 1)
    >>>
    >>> # plot with traffic estimations
    >>> colors = [edge[2]['traffic'] for edge in G.edges(data=True)]
    >>> widths = [edge[2]['traffic'] / 100 for edge in G.edges(data=True)]
    >>> node_sizes = [G.nodes[node]["population"] / 10 for node in G.nodes]
    >>> pos = nx.spring_layout(G)
    >>> nx.draw(G, pos, edge_color=colors, edge_cmap=plt.cm.Spectral_r, edge_vmin=min(colors),
    >>> edge_vmax=max(colors), node_size=node_sizes, node_color='darkgrey', with_labels=True,
    >>> width=widths)
    
    References:
    -----------
    ..[1] Ren Y, Ercsey-Ravasz M, Wang P, González MC, Toroczkai Z. Predicting commuter flows in spatial networks using a 
    radiation model based on temporal ranges. Nat Commun. 2014 Nov 6;5:5347. doi: 10.1038/ncomms6347. PMID: 25373437.
    """

    fluxes = {}
    for i, j in G.edges():
        G[i][j]['traffic'] = 0
    
    for a in G.nodes:
        m_a = G.nodes[a][population]
        pred, costs = nx.shortest_paths.dijkstra_predecessor_and_distance(G, a, cutoff=C, weight=cost)
        mpt_nodes = [*pred.keys()]
        print (mpt_nodes)
        
        def distribute_phi(b_j, phi):
                """Distribute flux contribution recursively across predecessors."""
                n_pred = len(pred[b_j])
                if n_pred > 0:

                    if verbose:
                        print(f"distributing phi_({a},{b})={phi:.2f} for {b_j}'s {n_pred} predecessors: {pred[b_j]}.")
                    phi /= n_pred


                    for b_i in pred[b_j]:
                        G.edges[(b_i, b_j)]['traffic'] += phi
                        distribute_phi(b_i, phi)

        while len(mpt_nodes) > 1:
            # grab furthest away node
            b = mpt_nodes.pop()

            # calculate flux from a to b
            n_b = G.nodes[b][population]
            c_ab = costs[b]
            
            costs_ab = {key: value for key, value in costs.items() if key not in [a, b]}
            s_ab = sum([G.nodes[node][population] for node, c_v in costs_ab.items() if c_v <= c_ab])
            
            phi_ab = zeta * (m_a**2 * n_b) / ((m_a + s_ab) * (m_a + s_ab + n_b))
            fluxes[(a, b)] = phi_ab
            distribute_phi(b, phi_ab)
            
    return G, fluxes




def test_get_traffic():
    
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 4), (3, 4), (4, 5), (1, 3)])
    for node in G.nodes:
        G.nodes[node]["population"] = 1

    # pos = nx.spring_layout(G)
    # nx.draw(G, with_labels=True)

    G, fluxes = get_traffic(G, 1000, None, 1, verbose=False)
    print (G.edges)
    assert fluxes[4, 5] == 1 / 12
    assert fluxes[5, 4] == 1 / 2
    assert fluxes[2, 1] == 1 / 6
    assert fluxes[2, 4] == 1 / 6
    assert fluxes[4, 2] == 1 / 12
    assert fluxes[1, 4] == 1 / 12
    assert fluxes[4, 1] == 1 / 20
    assert fluxes[2, 5] == 1 / 20
    assert fluxes[5, 2] == 1 / 12
    assert fluxes[1, 5] == 1 / 20
    assert fluxes[5, 1] == 1 / 20


    t_23 = fluxes[2,4] + fluxes[4, 2] + fluxes[2,5] + fluxes[5,2] + 0.5 * (fluxes[1,4] + fluxes[4,1] + fluxes[1,5] + fluxes[5,1] + fluxes[2, 3] + fluxes[3,2])
    assert np.isclose(t_23, G.edges[(2, 4)]['traffic'])

if __name__ == '__main__':
    test_get_traffic()
