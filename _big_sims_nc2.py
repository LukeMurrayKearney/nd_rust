import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import json

n, iters = 100_000, 20

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

data = 'comix2'
models = ['sbm', 'dpln']
scale = 'none'
# ## Attempt 1
# taus = np.array([0.002,0.004,0.006,0.008,0.01,0.02,0.03,0.04,0.05,0.06,0.07])
# ## attempt 2 
# taus = np.array([0.001,0.003,0.005,0.007,0.009,0.015,0.025,0.035,0.045,0.055,0.065])
## attempt 3 
# taus = np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065])
## attempt 4
# taus = np.array([0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2])
## attempt 5
# taus = np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,
#                  0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2])
## attempt 6
# taus = [np.arange(0.21,0.3,0.01), np.arange(0.0005,0.0055,0.0005)]
## attempt 7
# taus = [np.concatenate((np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,
#                  0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]), np.arange(0.21,0.3,0.01))),
#         np.concatenate((np.arange(0.0005,0.0055,0.0005), np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,
#                  0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2])))]
## attempt 9
# taus = [np.arange(0.05,0.005,0.3), np.arange(0.00025,0.015,0.00025)]
## no age 2
# taus = np.arange(0.00025,0.015,0.00025)
## no age 3
taus = np.arange(0.001, 0.02, 0.002)


# for i, model in enumerate(models): 
#     contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
#     if model == models[0]:
#         params = []
#     else:
#         params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
#     results = nd_p.taus_sims(taus=taus[i], partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
#     with open(f'../output_data/simulations/big/right_{data}_{model}9.json', 'w') as file:
#         json.dump(results, file)
    
# no age

buckets = np.array([])
partitions = [n]

egos, contact_matrix, params = nd_p.fit_to_data(input_file_path=f'input_data/{data}.csv',dist_type=models[1], buckets=buckets,save_fig=False)
print('starting sims')
results = nd_p.taus_sims(taus=taus, partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=models[1], prop_infec=10/n, scaling=scale)
with open(f'../output_data/simulations/big/noAge_{data}_{models[1]}3.json', 'w') as file:
    json.dump(results, file)