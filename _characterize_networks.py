import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import networkx as nx
import json

n, iters = 100_000, 1_000

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['poly','comix1', 'comix2']
models = ['sbm', 'nbinom', 'dpln']
scale1 = 'fit2'

for data in datas:
    for model in models:
        print(f'{data}: {model}')
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == 'sbm':
            params=[]
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        network = nd_p.build_network(n,partitions,contact_matrix,params,model)
        G = nd_p.to_networkx(network=network)
        # Compute the average degree
        degrees = dict(G.degree())
        average_degree = sum(degrees.values()) / float(G.number_of_nodes())

        # Compute the degree assortativity coefficient
        degree_assortativity = nx.degree_assortativity_coefficient(G)

        # Compute the average clustering coefficient
        average_clustering = nx.average_clustering(G)

        # Print the results
        print("what it says:", np.mean(network['degrees']))
        print("Average Degree:", average_degree)
        print("Degree Assortativity Coefficient:", degree_assortativity)
        print("Average Clustering Coefficient:", average_clustering)
