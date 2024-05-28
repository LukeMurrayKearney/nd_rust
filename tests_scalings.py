import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import json

n, iters = 100_000, 100

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

data_names = ['comix1','comix2', 'poly']
models = ['sbm','nbinom','dpln']
scalings = ['log','sqrt','linear']
taus = []
r0 = 3

for scale in scalings:
    for data in data_names:
        for model in models: 
            print(f'{data}: {model}', flush=True)
            contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
            if model == 'sbm':
                params = []
            else:
                params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
            
            # find suitable tau for r0
            result = nd_p.fit_to_r0(partitions=partitions, network_params=params, contact_matrix=contact_matrix, iterations=30, n=n, prop_infec=10/n, r0=r0, dist_type=model, scaling=scale)
            # simulate outbreaks with r0
            infections = nd_p.simulate(partitions=partitions, contact_matrix=contact_matrix,network_params=params, tau=result['tau'], iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
            # save outputs
            with open(f'output_data/simulations/sims_fit_{r0}_{data}_{model}_{scale}.json', 'w') as file:
                json.dump(infections, file)
            # save tau used for comix1 comparison
            if data == "comix2":
                taus.append(result['tau'])

    data = data_names[0]
    for i, model in enumerate(models):
        print(f'{data}: {model}', flush=True)
        contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_{data}.csv', delimiter=',')
        if model == 'sbm':
            params = []
        else:
            params = np.genfromtxt(f'input_data/parameters/params_{data}_{model}.csv', delimiter=',')
        infections = nd_p.simulate(partitions=partitions, contact_matrix=contact_matrix,network_params=params, tau=taus[i], iterations=iters, n=n, dist_type=model, prop_infec=10/n, scaling=scale)
        # save outputs
        with open(f'output_data/simulations/sims_compare_{r0}_{data}_{model}_{scale}.json', 'w') as file:
            json.dump(infections, file)
