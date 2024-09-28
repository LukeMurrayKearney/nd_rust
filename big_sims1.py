import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import json

n, iters = 100_000, 100

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

data = 'comix1'
models = ['sbm', 'nbinom', 'dpln']
scale = 'fit1'
taus = np.arange(0.01,0.31, 0.01)

for model in models: 
    contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
    if model == models[0]:
        params = []
    else:
        params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
    results = nd_p.taus_sims(taus=taus, partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
    with open(f'../output_data/simulations/big/{data}_{model}.json', 'w') as file:
        json.dump(results, file)
    