import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import pandas as pd
import json

n, iters = 100_000, 100

buckets = np.array([])
partitions = [n]

data_names = ['comix1','comix2', 'poly']
model ='dpln'
scalings = ['none','fit1']

r0 = 3


for scale in scalings:
    for data in data_names:
        print(f'{scale} {data}')
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        old_partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]
        #get mean contacts
        contact_matrix = sum([sum(row)*old_partitions[i] for i, row in enumerate(contact_matrix)])/n
        #read in data and fit 
        _, check, params = nd_p.fit_to_data(nd_p.read_in_dataframe(f'input_data/{data}.csv'), dist_type=model,buckets=buckets,save_fig=False,log=False)
        print(check, contact_matrix)
        
        # find suitable tau for r0
        result = nd_p.fit_to_r0(partitions=partitions, network_params=params, contact_matrix=contact_matrix, iterations=iters, n=n, prop_infec=10/n, r0=r0, dist_type=model, scaling=scale)
        # simulate outbreaks with r0
        infections = nd_p.simulate(partitions=partitions, contact_matrix=contact_matrix,network_params=params, tau=result['tau'], iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
        # save outputs
        with open(f'../output_data/simulations/new/{data}_{scale}.json', 'w') as file:
            json.dump(infections, file)
