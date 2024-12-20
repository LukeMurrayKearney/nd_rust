import nd_rust as nd_r
import nd_python as nd_p 
import numpy as np
import csv
import json

n, iters = 100_000, 30

buckets = np.array([5,12,18,30,40,50,60,70])
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['poly','comix1', 'comix2']
models = ['sbm', 'nbinom', 'dpln']

# distance matrix of EMD
bins = np.arange(0,len(partitions),1)
distance_matrix = np.zeros((len(partitions), len(partitions)))
for i in bins:
    for j in bins:
        distance_matrix[i,j] = np.float64(np.abs(i-j))

for data in datas:
    for model in models:
        print(data, model)
        error, error_breakdown = [], []
        error_with_itself, error_with_itself_breakdown = [], []
        egos, contact_matrix, params = nd_p.fit_to_data(input_file_path=f'input_data/{data}.csv',dist_type=model)
        for i in range(iters):
            if i % 2 == 0:
                print(i)

            # my model error
            network = nd_p.build_network(n,partitions,contact_matrix,params,model)
            # errors, err_pp = nd_p.emd_error(egos, network, distance_matrix=distance_matrix)
            # error_breakdown.append(errors)
            # error.append(err_pp)
            
            
            # network error of my model with true network
            data = nd_p.data_from_network(network=network)
            egos_itself, contact_matrix_itself, params_itself = nd_p.fit_to_data(df=data, save_fig=False, output_file_path="fits/network_comix1", buckets=buckets,dist_type=model)
            network = nd_p.build_network(n, partitions, params_itself, contact_matrix_itself,dist_type=model)
            errors, err_pp = nd_p.emd_error(egos_itself, network, distance_matrix=distance_matrix)
            error_with_itself_breakdown.append(errors)
            error_with_itself.append(err_pp)
            
        # with open(f'../output_data/errors/breakdown_{data}_{model}.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for row in error_breakdown:
        #         writer.writerow(row)
                
        with open(f'../output_data/errors/breakdown_itself_{data}_{model}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in error_with_itself_breakdown:
                writer.writerow(row)

        # with open(f'../output_data/errors/{data}_{model}.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(error)


        with open(f'../output_data/errors/itself_{data}_{model}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(error_with_itself)
            
        