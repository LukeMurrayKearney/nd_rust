import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import json
import pandas as pd

n, iters = 100_000, 100

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

models = ['sbm','nbinom','dpln']

contact_matrix = np.genfromtxt(f'input_data/contact_matrices/contact_matrix_poly.csv', delimiter=',')

params = np.genfromtxt(f'input_data/parameters/params_poly_{models[2]}.csv', delimiter=',')

df_outbreak = pd.read_json('input_data/fit_sims/influenza_hospitalisations.json')

data = [float(x) for x in df_outbreak['hospitalisations']]

days = list(range(6,len(df_outbreak['hospitalisations'])*7,7))


results = nd_p.mcmc(data=data, days=days, partitions=partitions, contact_matrix=contact_matrix,network_params=[], outbreak_params=[0.1,5], dist_type=models[2], iters=10_000, prior_param=50, n=n)

np.savetxt(f'../output_data/fit_to_data/{models[2]}_mcmc_taus.csv', np.array(results['taus']), delimiter=',')

        