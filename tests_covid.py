import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import json

n, iters = 100_000, 1_000

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

data1, data2 = 'comix1', 'comix2'
model ='dpln'
scale = 'fit1'

#taken from 2 week delay of SPI-M-0 data
r0 = 1.04167

print('CoMix1')
contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data1}.csv', delimiter=',')
params = np.genfromtxt(f'input_data/parameters/params_{data1}_{model}.csv', delimiter=',')

# find suitable tau for r0
result = nd_p.fit_to_r0(partitions=partitions, network_params=params, contact_matrix=contact_matrix, iterations=iters, n=n, prop_infec=10/n, r0=r0, dist_type=model, scaling=scale, num_networks=100,num_restarts=100)
# simulate outbreaks with r0
infections = nd_p.simulate(partitions=partitions, contact_matrix=contact_matrix,network_params=params, tau=result['tau'], iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
# save outputs
with open(f'../output_data/simulations/covid/{data1}.json', 'w') as file:
    json.dump(infections, file)
# save tau used for comix2 comparison
tau = result['tau']

print('CoMix2')
contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data2}.csv', delimiter=',')
params = np.genfromtxt(f'input_data/parameters/params_{data2}_{model}.csv', delimiter=',')
infections = nd_p.simulate(partitions=partitions, contact_matrix=contact_matrix,network_params=params, tau=tau, iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
# save outputs
with open(f'../output_data/simulations/covid/{data2}.json', 'w') as file:
    json.dump(infections, file)