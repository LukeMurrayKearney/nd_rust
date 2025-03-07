import nd_rust as nd_r
import nd_python_avon as nd_p 
import numpy as np
import json

n, iters = 100_000, 48

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

data = 'poly'
models = ['sbm', 'dpln']
scale = 'none'

## test
taus = [np.arange(0.002,0.2,0.001)/10, np.arange(0.002,0.8,0.001)/10]

## a1
# taus = [np.arange(0.002,0.2,0.001),np.arange(0.002,0.8,0.001)]


## attempt 1 
# taus = np.array([0.02,0.03,0.04,0.05,0.06])
## attempt 2
# taus = np.array([0.005,0.01,0.015,0.025,0.035,0.045,0.055,0.065,0.07,0.075,0.08,0.02,0.03,0.04,0.05,0.06])
## attempt 3
# taus = np.array([0.005,0.01,0.015,0.025,0.035,0.045,0.055,0.065,0.07,0.075,0.08,0.02,0.03,0.04,0.05,0.06])
## attempt 4
# taus = [np.array([0.0025,0.0075,0.085,0.09,0.095,0.1]), np.arange(0.001,0.01,0.001)]
## attempt 7
# taus = [np.concatenate((np.array([0.005,0.01,0.015,0.025,0.035,0.045,0.055,0.065,0.07,0.075,0.08,0.02,0.03,0.04,0.05,0.06]), np.array([0.0025,0.0075,0.085,0.09,0.095,0.1]))),
#         np.concatenate((np.arange(0.001,0.01,0.001), np.array([0.005,0.01,0.015,0.025,0.035,0.045,0.055,0.065,0.07,0.075,0.08,0.02,0.03,0.04,0.05,0.06])))]
## attempt 9 
# taus = [np.arange(0.002,0.1,0.002),np.arange(0.002,0.6,0.002)]
## no age 2
# taus = np.arange(0.002,0.6,0.004)
## no age 3,4
# taus = np.arange(0.005,0.05,0.005)
## attempt 10
# taus = [np.arange(0.002,0.1,0.001),np.arange(0.002,0.6,0.001)]


for i, model in enumerate(models): 
    contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
    if model == models[0]:
        params = []
    else:
        params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
    results = nd_p.taus_sims(taus=taus[i], partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale, inv_gamma=40)
    with open(f'../output_data/simulations/big/fixed/{data}_{model}_test.json', 'w') as file:
        json.dump(results, file)
    
# # no age

# buckets = np.array([])
# partitions = [n]

# contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}_no_age.csv', delimiter=',')
# params = np.genfromtxt(f'input_data/parameters/params_{data}_{models[1]}_no_age.csv', delimiter=',')
# # egos, contact_matrix, params = nd_p.fit_to_data(input_file_path=f'input_data/{data}.csv',dist_type=models[1], buckets=buckets,save_fig=False)
# results = nd_p.taus_sims(taus=taus, partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=models[1], prop_infec=10/n, scaling=scale)
# with open(f'../output_data/simulations/big/noAge_{data}_{models[1]}4.json', 'w') as file:
#     json.dump(results, file)