import networkx as nx
from edgesFinal import edgesFinal
from networkx.algorithms.community.centrality import girvan_newman

def graph_analysis(edgesFinal):
    G = nx.Graph()
    G.add_weighted_edges_from(edgesFinal)
    
    print(G)
    print('Density: ', nx.density(G))
    print('Diameter: ', nx.diameter(G))

    eigenvector = nx.eigenvector_centrality(G, tol=1.0e-3,  weight='weight')
    print('Eigenvector: ')
    print(sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[0:80])

    betweenness = nx.betweenness_centrality(G, normalized=False,  weight='weight')
    print('Betweenness: ')
    print(sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[0:80])

    closeness = nx.closeness_centrality(G, distance='weight')
    print('Closeness:')
    sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
    print(sorted_closeness[:80])

    pr = nx.pagerank(G, alpha=0.85,  weight='weight')
    print('Pagerank: ')
    print(sorted(pr.items(), key=lambda x: x[1], reverse=True)[0:80])

    print('Eccentricity. ' )
    print(nx.eccentricity(G, weight='weight'))

    print('Center nodes: ')
    print(nx.center(G, weight='weight'))


    communities = nx.community.louvain_communities(G,  seed=123, weight='weight')
    
    print('Comunities')
    print(communities)
graph_analysis(edgesFinal)