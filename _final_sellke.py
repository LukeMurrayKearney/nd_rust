import nd_python_avon as nd_p 
import numpy as np
import json
n, iters = 100_000, 48

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['comix1','comix2','poly']
models = ['sbm','dpln']
scales = ['none', 'none']

## 0
# taus = [[np.arange(0.001,0.1,0.0025), np.arange(0.001,0.06,0.001)],
#         [np.arange(0.001,0.1,0.0025), np.arange(0.001,0.06,0.001)],
#         [np.arange(0.001,0.1,0.0025), np.arange(0.001,0.1,0.0025)]]
## 1,2,3,4
# taus = [[np.arange(0.001,0.1,0.005), np.arange(0.001,0.06,0.002)],
#         [np.arange(0.001,0.1,0.005), np.arange(0.001,0.06,0.002)],
#         [np.arange(0.001,0.1,0.005), np.arange(0.001,0.1,0.005)]]
# 5,6,7,8,9
# taus = [[np.arange(0.1,0.205,0.005), np.arange(0.0005,0.01,0.0005)],
#         [np.arange(0.05,0.2,0.01), np.arange(0.0005,0.01,0.0005)],
#         [np.arange(0.001,0.1,0.005), np.arange(0.0025,0.05,0.0025)]]
# 10,11,12,13,14,15
taus = [[np.arange(0.1,0.305,0.005), np.arange(0.0001,0.005,0.0001)],
        [np.arange(0.05,0.25,0.005), np.arange(0.0001,0.005,0.0001)],
        [np.arange(0.001,0.1,0.002), np.arange(0.0001,0.04,0.0001)]]

### long
## 0,1,2
# taus = [[np.arange(0.001,0.1,0.005), np.arange(0.001,0.06,0.002)],
#         [np.arange(0.001,0.1,0.005), np.arange(0.001,0.06,0.002)],
#         [np.arange(0.001,0.1,0.005), np.arange(0.001,0.1,0.005)]]

for i, data in enumerate(datas):
    for j, model in enumerate(models):
        print(data, model)
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == 'sbm':
            params = []
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        result = nd_p.big_sellke_sims(partitions=partitions,contact_matrix=contact_matrix,network_params=params,n=n,dist_type=model,num_networks=2,iterations=iters, taus=taus[i][j],prop_infec=5/n, scaling=scales[j])
        with open(f'../output_data/simulations/big/sellke/SIR/13_{data}_{model}_{scales[j]}.json','w') as f:
            json.dump(result, f)
print('done')
