import nd_rust as nd_r
import nd_python_avon as nd_p 
import numpy as np
import json

n, iters = 100_000, 48

buckets = np.array([])
partitions = [n]

data = 'comix1'
model = 'dpln'
## attempt 1
taus = np.arange(0.05,0.55,0.05)

scale = 'fit1'
contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}_no_age.csv', delimiter=',')
params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}_no_age.csv', delimiter='\n')
print(len(params))
# egos, contact_matrix, params = nd_p.fit_to_data(input_file_path=f'input_data/{data}.csv',dist_type=model, buckets=buckets,save_fig=False)
results = nd_p.taus_sims(taus=taus, partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
with open(f'../output_data/simulations/big/{data}_{model}_{scale}_no_age.json', 'w') as file:
    json.dump(results, file)