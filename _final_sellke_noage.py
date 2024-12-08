import nd_python_avon as nd_p 
import numpy as np
import json

n, iters = 100_000, 48

buckets = np.array([])
partitions = [n]


# datas = ['comix1','comix2','poly']
datas = ['comix1']
# datas = ['comix2']
# datas = ['poly']

models = ['dpln']

# scales = ['none']
# scales = ['fit1']
scales = ['fit2']

## 0
# taus = [[np.arange(0.001,0.07,0.002)]]
# # taus = [[np.arange(0.001,0.07,0.002)]]
# # taus = [[np.arange(0.001,0.05,0.002)]]
# taus = [[10*x for x in a] for a in taus]
## 1,2
# taus = [[np.arange(0.001,0.07,0.001)]]
# taus = [[np.arange(0.001,0.07,0.002)]]
# taus = [[np.arange(0.001,0.05,0.002)]]
# taus = [[20*x for x in a] for a in taus]
## 3,4
# taus = [[np.arange(0.001,0.07,0.001)]]
# taus = [[np.arange(0.001,0.07,0.002)]]
# taus = [[np.arange(0.001,0.05,0.002)]]
# taus = [[30*x for x in a] for a in taus]
## 5, 6 (7)
taus = [[np.arange(0.5,20,0.25)]]
# taus = [[np.arange(0.1,2,0.05)]]
# taus = [[np.arange(0.025,2.5,0.025)]]


for i, data in enumerate(datas):
    for j, model in enumerate(models):
        print(data, model)
        # contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        # if model == 'sbm':
        #     params = []
        # else:
        #     params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        _, contact_matrix, params = nd_p.fit_to_data(f'input_data/{data}.csv',dist_type=model, buckets=buckets, save_fig=False)
        result = nd_p.big_sellke_sims(partitions=partitions,contact_matrix=contact_matrix,network_params=params,n=n,dist_type=model,num_networks=1,iterations=iters, taus=taus[i][j],prop_infec=5/n, scaling=scales[j])
        with open(f'../output_data/simulations/big/sellke/SIR/7_{data}_{model}_{scales[j]}_noage.json','w') as f:
            json.dump(result, f)
print('done')
