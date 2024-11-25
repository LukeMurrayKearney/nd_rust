# import os
import json
import numpy as np

# # Specify the directory you want to loop through
# directory = '../output_data/simulations/big/sellke/'  # Replace with your directory path

# # Loop through all files and directories in the specified directory
# for filename in os.listdir(directory):
#     # Construct full file path
#     filepath = os.path.join(directory, filename)
#     print(filepath)
#     try:
#         with open(filepath,'r') as f:
#             result = json.load(f)
#             f.close()
#         reduced = {'SIR': result['SIR'], 't': result['t'], 'secondary_cases': result['secondary_cases'], 'generations': result['generations']}
#         print(filepath[:-5] + '_red.json')
#         with open(filepath[:-5] + '_red.json','w') as f:
#             json.dump(reduced,f)
#             f.close()
#     except Exception as e: 
#         print(f"Error processing file {filepath}: {e}")
#         continue

datas = ['comix1', 'comix2','poly']
models = ['sbm', 'dpln']
for sim_num in [3,4,5,6,7,8,9,10]:
    final_sizes, peak_heights, r0s = {}, {}, {}
    for data in datas:
        final_sizes[data], peak_heights[data], r0s[data] = {}, {}, {}
        for model in models:
            print(data, model)
            final_sizes[data][model], peak_heights[data][model], r0s[data][model] = [], [], []
            for i in range(20):
                final_sizes[data][model].append([])
                peak_heights[data][model].append([])
                r0s[data][model].append([])
                try:
                    with open(f'../output_data/simulations/big/sellke/{data}_{model}_{i}_{sim_num}.json', 'r') as f:
                        tmp = json.load(f)
                    for j, sir in enumerate(tmp['SIR']):
                        final_sizes[data][model][i].append(sir[-1][-1])
                        peak_heights[data][model][i].append(np.max([a[1] for a in sir]))
                        tmp_r0 = []
                        max_gen = max(tmp['generations'][j])
                        for node_idx, cases in enumerate(tmp['secondary_cases'][j]):
                            if tmp['generations'][j][node_idx] in [1] and max_gen >= 4:
                                tmp_r0.append(cases)
                        r0s[data][model][i].append(np.mean(tmp_r0))                
                except:
                    print(f'we have no file for {data}, {model} ({i})')

    for data in datas:                
        with open(f'../output_data/simulations/big/sellke/SIR/summary_stats/try1_fs_{data}_{sim_num}.json', 'w') as f:
            json.dump(final_sizes[data], f)
        for key in peak_heights[data].keys():
            peak_heights[data][key] = [[int(y) for y in x] for x in peak_heights[data][key]] 
        with open(f'../output_data/simulations/big/sellke/SIR/summary_stats/try1_ph_{data}_{sim_num}.json', 'w') as f:
            json.dump(peak_heights[data], f)
        with open(f'../output_data/simulations/big/sellke/SIR/summary_stats/try1_r0_{data}_{sim_num}.json', 'w') as f:
            json.dump(r0s[data], f)
