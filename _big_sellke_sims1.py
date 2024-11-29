import nd_python_avon as nd_p 
import numpy as np
import json
n, iters = 100_000, 48

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['comix1','comix2','poly']
models = ['sbm','dpln']
scales = ['none']

## 0-11
k_hat = 6
taus = [i/(7*k_hat*2) for i in range(1,21)]
## 12-23
# k_hat = 6
# taus = [i/(7*k_hat) for i in range(1,21)]
## 24-35
# k_hat = 6
# taus = [i/(7*k_hat*10) for i in range(1,21)]
## 36-47
# k_hat = 6
# taus = [i/(7*k_hat*100) for i in range(1,21)]
# ## 48-59
# k_hat = 6
# taus = [i/(7*k_hat*10) for i in range(1,21)]
## 60-71
# taus = [i/(7*k_hat*50) for i in range(1,21)]
## 72-83
# taus = [(i/7*k_hat*50) for i in range(1,21)]
## 84-90
# taus = [(2*i/7*k_hat) for i in range(1,21)]
## 91-97
# taus = [(2*i/7*k_hat) for i in range(1,21)]

#### new ####
## 0-11
# k_hat = 6
# taus = [i/(7*k_hat*2) for i in range(1,21)]
## 12-23
# taus = [i/(7*k_hat) for i in range(1,21)]
## 23-30
taus = [2*i/(7*k_hat) for i in range(1,21)]
## 31-39
# taus = [2*i/(7*k_hat) for i in range(1,21)]


#### small #####
## 0-100
taus = np.linspace(0.0001,1,100)


for sim_num in range(11,21):
    for i, data in enumerate(datas):
        for model in models:
            contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
            if model == 'sbm':
                params = []
            else:
                params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
            for index, tau in enumerate(taus):
                result = nd_p.sellke_test(partitions=partitions,contact_matrix=contact_matrix,network_params=params,n=n,dist_type=model,iterations=iters, tau=tau,prop_infec=10/n)
                with open(f'../output_data/simulations/big/sellke/{data}_{model}_{index}_{sim_num}_small.json','w') as f:
                    json.dump(result, f)
