import nd_python_avon as nd_p 
import numpy as np
import json
n, iters = 100_000, 48

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['comix1','comix2','poly']
models = ['sbm','dpln']
scales = ['fit1', 'fit2']

## 0
# taus = [[np.arange(0.001,0.1,0.0025), np.arange(0.001,0.06,0.001)],
#         [np.arange(0.001,0.1,0.0025), np.arange(0.001,0.06,0.001)],
#         [np.arange(0.001,0.1,0.0025), np.arange(0.001,0.1,0.0025)]]
## 1,2,3,4
taus = [[np.arange(0.001,0.1,0.005), np.arange(0.001,0.06,0.002)],
        [np.arange(0.001,0.1,0.005), np.arange(0.001,0.06,0.002)],
        [np.arange(0.001,0.1,0.005), np.arange(0.001,0.1,0.005)]]
taus = [[10*x for x in a] for a in taus]

for i, data in enumerate(datas):
    for j, model in enumerate(models):
        print(data, model)
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == 'sbm':
            params = []
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        result = nd_p.big_sellke_sims(partitions=partitions,contact_matrix=contact_matrix,network_params=params,n=n,dist_type=model,num_networks=1,iterations=iters, taus=taus[i][j],prop_infec=10/n, scaling=scales[j])
        with open(f'../output_data/simulations/big/sellke/SIR/4_{data}_{model}_{scales[j]}.json','w') as f:
            json.dump(result, f)
print('done')