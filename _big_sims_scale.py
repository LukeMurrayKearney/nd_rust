import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import json

n, iters = 100_000, 100

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['comix2','comix1']
model = 'dpln'
## attempt 1
# taus = np.array([0.2,0.21,0.22,0.23,0.24,0.25,0.3,0.4,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5])
## attempt 2
# taus = np.array([0.1 + 0.01*a for a in range(10)])
## attempt 3
# taus = np.array([0.2,0.21,0.22,0.23,0.24,0.25,0.3,0.4,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,
#                  0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.52,0.54,0.56,0.58,0.6,0.65,0.7,0.8,0.85,0.9,0.95,1])
## attempt 4
# taus = np.array([0.2,0.21,0.22,0.23,0.24,0.25,0.3,0.4,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,
#                  0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.52,0.54,0.56,0.58,0.6,0.65,0.7,0.8,0.85,0.9,0.95,1])
## attempt 5
# taus = np.arange(0.01,0.1,0.005)
## attempt 7 
# taus = [np.concatenate((np.arange(0.01,0.1,0.005), np.array([0.2,0.21,0.22,0.23,0.24,0.25,0.3,0.4,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,
#                   0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.52,0.54,0.56,0.58,0.6,0.65,0.7,0.8,0.85,0.9,0.95,1]))),
#         np.concatenate((np.arange(0.01,0.1,0.005), np.array([0.2,0.21,0.22,0.23,0.24,0.25,0.3,0.4,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,
#                   0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.52,0.54,0.56,0.58,0.6,0.65,0.7,0.8,0.85,0.9,0.95,1])))]
## attempt 8
# taus = np.arange(0.001,0.026,0.001)
## attempt 9
taus = 

print(1)
data = datas[0]
scale = 'fit2'
contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
results = nd_p.taus_sims(taus=taus, partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
with open(f'../output_data/simulations/big/age_scale_{data}_{model}_{scale}8.json', 'w') as file:
    json.dump(results, file)
    
print(2)
data = datas[1]
scale = 'fit1'
contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
results = nd_p.taus_sims(taus=taus, partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
with open(f'../output_data/simulations/big/age_scale_{data}_{model}_{scale}8.json', 'w') as file:
    json.dump(results, file)