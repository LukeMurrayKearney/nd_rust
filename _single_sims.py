import nd_python as nd_p 
import numpy as np
import json

n, iters = 10_000, 1

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['comix1', 'comix2', 'poly']
# datas = ['comix1']
models = ['sbm', 'dpln']
scales = ['none', 'none']
# scales = ['fit1', 'fit2']
## r0 == 3
taus = [0.19, 0.022, 0.14, 0.00375, 0.055, 0.034]
# taus = [0.53, 0.0095]
taus = [a/10 for a in taus]

count = 0
for i, data in enumerate(datas[:-1]):
    for model in models:
        print(count)
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == 'sbm':
            params = []
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        results = nd_p.simulate(partitions=partitions, contact_matrix=contact_matrix, network_params=params, tau=taus[count], iterations=iters, n=n,dist_type=model, inv_gamma=40, prop_infec=10/n,scaling=scales[i])
        with open(f'../output_data/networks/with_outbreaks/{data}_{model}_{scales[i]}_long.json', 'w') as file:
            json.dump(results, file)
        count += 1