import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import json

n, iters = 100_000, 100

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['comix2','comix1','poly']
models = ['sbm', 'nbinom', 'dpln']

for data in datas:
    for model in models:
        print(data,model)
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == models[0]:
            params = []
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        network = nd_p.build_network(n,partitions,contact_matrix,params,model)
        with open(f'..output_data/networks/{data}_{model}.json', 'w') as f:
            json.dump(network, f)