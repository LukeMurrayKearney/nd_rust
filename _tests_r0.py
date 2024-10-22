import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import json

n, iters = 100_000, 30

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

data_names = ['comix1','comix2', 'poly']
# models = ['sbm','nbinom','dpln']
models = ['sbm']
scale = ['none' for _ in range(data_names)]
# sbm
# taus = [0.19,0.14,0.055]
# dpln
taus = [0.022, 0.00375, 0.034]
# scaled
# taus = [0.53, 0.00975]
# scales = ['fit1', 'fit2']

# r0 = 3
for i, data in enumerate(data_names):
    for j, model in enumerate(models): 
        print(f'{data}: {model}', flush=True)
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == 'sbm':
            params = []
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        
        # find suitable tau for r0
        # result = nd_p.fit_to_r0(partitions=partitions, network_params=params, contact_matrix=contact_matrix, iterations=30, n=n, prop_infec=10/n, dist_type=model)
        # simulate outbreaks with r0
        infections = nd_p.simulate(partitions=partitions, contact_matrix=contact_matrix,network_params=params, tau=taus[i], iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scales[i])
        # save outputs
        with open(f'../output_data/simulations/test_sims_{data}_{model}.json', 'w') as file:
            json.dump(infections, file)
