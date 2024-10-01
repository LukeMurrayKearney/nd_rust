import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import json

n, iters = 100_000, 30

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['comix2','comix1']
model = 'dpln'
## attempt 1
# taus = np.array([0.2,0.21,0.22,0.23,0.24,0.25,0.3,0.4,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5])
## attempt 2
taus = np.array([0.1 + 0.01*a for a in range(10)])

print(1)
data = datas[0]
scale = 'fit2'
contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
results = nd_p.taus_sims(taus=taus, partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
with open(f'../output_data/simulations/big/right_{data}_{model}_{scale}2.json', 'w') as file:
    json.dump(results, file)
    
print(2)
data = datas[1]
scale = 'fit1'
contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
results = nd_p.taus_sims(taus=taus, partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
with open(f'../output_data/simulations/big/right_{data}_{model}_{scale}2.json', 'w') as file:
    json.dump(results, file)