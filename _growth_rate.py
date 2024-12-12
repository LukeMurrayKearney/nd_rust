import nd_python_avon as nd_p 
import numpy as np
import pandas as pd
import json

n, iters = 100_000, 48
buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['comix1','comix2','poly']
models = ['sbm']
scales = ['none']

## 0,1,2,3
taus = [[np.arange(0.01,0.11,0.01)],
        [np.arange(0.01,0.11,0.01)],
        [np.arange(0.01,0.11,0.01)]]
## 4,5
taus = [[np.arange(0.11,0.21,0.01)],
        [np.arange(0.11,0.21,0.01)],
        [np.arange(0.11,0.21,0.01)]]
## 6,7
taus = [[np.arange(0.21,0.31,0.01)],
        [np.arange(0.21,0.31,0.01)],
        [np.arange(0.21,0.31,0.01)]]
# 8,9
taus = [[np.arange(0.01,0.31,0.01)],
        [np.arange(0.01,0.31,0.01)],
        [np.arange(0.01,0.31,0.01)]]
# 10
taus = [[np.arange(0.005,0.41,0.005)],
        [np.arange(0.01,0.41,0.01)],
        [np.arange(0.01,0.41,0.01)]]

for i, data in enumerate(datas):
    for j, model in enumerate(models):
        print(data, model)
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == 'sbm':
            params = []
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        result = nd_p.sellke_sims_growth_rate(partitions=partitions,contact_matrix=contact_matrix,network_params=params,n=n,dist_type=model,num_networks=1,iterations=iters, taus=taus[i][j],prop_infec=5/n, scaling=scales[j])
        with open(f'../output_data/simulations/big/sellke/growth_rate/10_{data}_{model}.json','w') as f:
            json.dump(result, f)
print('done')
