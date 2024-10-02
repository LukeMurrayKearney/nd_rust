import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import networkx as nx
import json

ns, iters = np.arange(10_000,210_000,10_000), 100
ns = [100_000]

buckets = np.array([5,12,18,30,40,50,60,70])


datas = ['poly','comix1', 'comix2']
models = ['sbm', 'nbinom', 'dpln']
scale1 = 'none'

for data in datas:
    for model in models:
        degrees, assort, clustering, diameters, short_paths = np.zeros((len(ns), iters)), np.zeros((len(ns), iters)), np.zeros((len(ns), iters)), np.zeros((len(ns), iters)), np.zeros((len(ns), iters))
        print(f'{data}: {model}')
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == 'sbm':
            params=[]
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        for i, n in enumerate(ns):
            print(n)
            partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]
            for j in range(iters):
                
                network = nd_p.build_network(n,partitions,contact_matrix,params,model)
                G = nd_p.to_networkx(network=network)
                # Compute the average degree
                degrees_dict = dict(G.degree())
                average_degree = sum(degrees_dict.values()) / float(G.number_of_nodes())

                # Compute the degree assortativity coefficient
                degree_assortativity = nx.degree_assortativity_coefficient(G)

                # Compute the average clustering coefficient
                average_clustering = nx.average_clustering(G)
                
                if nx.is_connected(G):
                    diameter = nx.diameter(G)
                    short_path = nx.average_shortest_path_length(G)
                else:
                    diameter = -1
                    short_path = -1

                degrees[i,j] = average_degree; assort[i,j] = degree_assortativity; clustering[i,j] = average_clustering; diameters[i,j] = diameter; short_paths[i,j] = short_path
        np.savetxt(f'../output_data/characterising/small_degrees_{data}_{model}.csv', degrees)
        np.savetxt(f'../output_data/characterising/small_assort_{data}_{model}.csv', assort)
        np.savetxt(f'../output_data/characterising/small_clust_{data}_{model}.csv', clustering)
        np.savetxt(f'../output_data/characterising/small_diameters_{data}_{model}.csv', diameters)
        np.savetxt(f'../output_data/characterising/small_path_{data}_{model}.csv', short_paths)
        