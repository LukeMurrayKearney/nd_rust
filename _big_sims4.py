import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import json

n, iters = 100_000, 100

buckets = np.array([])
partitions = [n]

datas = ['comix1', 'comix2', 'poly']
model = 'dpln'
scale = 'none'
taus = np.arange(0.01,0.31, 0.01)

for data in datas: 
    _, contact_matrix, params = nd_p.fit_to_data(nd_p.read_in_dataframe(f'input_data/{data}.csv'), dist_type=model,buckets=buckets,save_fig=False,log=False)
    results = nd_p.taus_sims(taus=taus, partitions=partitions, contact_matrix=contact_matrix,network_params=params, iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
    with open(f'../output_data/simulations/big/noAge_{data}_{model}.json', 'w') as file:
        json.dump(results, file)
    