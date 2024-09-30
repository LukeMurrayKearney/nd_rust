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
        if scale == scalings[1]:
            if data == data_names[1]:
                scale = 'fit2'
            elif data == data_names[2]:
              continue  
        print(f'{scale} {data}')
        _, contact_matrix, params = nd_p.fit_to_data(nd_p.read_in_dataframe(f'input_data/{data}.csv'), dist_type=model,buckets=buckets,save_fig=False,log=False)
        print(contact_matrix)
        
        # find suitable tau for r0
        result = nd_p.fit_to_r0(partitions=partitions, network_params=params, contact_matrix=contact_matrix, iterations=iters, n=n, prop_infec=10/n, r0=r0, dist_type=model, scaling=scale)
        # simulate outbreaks with r0
        infections = nd_p.simulate(partitions=partitions, contact_matrix=contact_matrix,network_params=params, tau=result['tau'], iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
        # save outputs
        with open(f'../output_data/simulations/new/noAge_{data}_{scale}.json', 'w') as file:
            json.dump(infections, file)
