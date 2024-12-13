import numpy as np
import matplotlib.pyplot as plt
import json
plt.rcParams.update({'font.size': 22})


n = 100_000
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n, 0.623*n, 0.759*n, 0.866*n, n]

datas = ['comix1','comix2','poly']
# datas = ['poly']
data_names = ['CoMix 1', 'CoMix 2', 'POLYMOD']
models = ['sbm']
model_names =['SBM']
scales = ['fit1', 'fit2']

## 0,1,2,3
taus = [[np.arange(0.01,0.11,0.01)],
        [np.arange(0.01,0.11,0.01)],
        [np.arange(0.01,0.11,0.01)]]

available_colors = [
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 
    'brown', 'black', 'white', 'gray', 'cyan', 'magenta', 'lime', 'teal'
]


# bins 
top, step = 5, 0.1
bins = np.arange(step, top + step, step)
bin_centers = np.array([a/2 if i == 0 else a - (a - bins[i-1])/2 for i,a in enumerate(bins)])
digit = [a for a in bins]; digit.append(1e6)
t_0 = 0
T_max = 14
max_gr, max_gr1 = 0, 0
for i, data in enumerate(datas):
    # if data != 'poly':
    xs,ys = [], []
    top, bot = [], []
    
    for j, model in enumerate(models):
        fs, ps = [[] for _ in bins], [[] for _ in bins]
        gr, gr1 = [], []
        r0s = []
        tau_res = []
        for sim_num in range(11):
            try:
                with open(f'../output_data/simulations/big/sellke/growth_rate/{sim_num}_{data}_{model}.json','r') as f:
                    tmp = json.load(f)
                sims_per_tau = int(len(tmp['t'])/len(taus[i][j]))
                for tau, tau_sim in enumerate(tmp['r0_1']):
                # for tau, tau_sim in enumerate(tmp['r0_23']):
                    
                    r0 = np.mean([a for a in tau_sim if a > 0]) if len([a for a in tau_sim if a > 0]) > 0 else -1
                    if r0 == -1: 
                        continue
                    I = [a for index, a in enumerate(tmp['sir'][tau*sims_per_tau : sims_per_tau*(tau+1)]) if tau_sim[index] > 0]
                    t = [[val for idx, val in enumerate(a) if idx == 0 or val > 0] for index, a in enumerate(tmp['t'][tau*sims_per_tau : sims_per_tau*(tau+1)]) if tau_sim[index] > 0]
                    final_sizes = [a for a in tmp['final_size'][tau] if a > 0]
                    peak_heights = [a for a in tmp['peak_height'][tau] if a > 0]
                    bin_idx = np.digitize(r0, digit, right=False)
                    if bin_idx < len(bins):
                        for k, sims in enumerate(final_sizes):
                            fs[bin_idx].append(sims/n)
                            ps[bin_idx].append(peak_heights[k]/n)
                    
                    r0s.append(r0)
                    tau_res.append(tau_sim)
                    gr.append({})
                    # gr1.append({})
                    ## r0 vs tau 
                    if sim_num in [4,5]:
                        ## 4,5
                        taus = [[np.arange(0.11,0.21,0.01)],
                                [np.arange(0.11,0.21,0.01)],
                                [np.arange(0.11,0.21,0.01)]]
                    if sim_num in [6,7]:
                        ## 6,7
                        taus = [[np.arange(0.21,0.31,0.01)],
                                [np.arange(0.21,0.31,0.01)],
                                [np.arange(0.21,0.31,0.01)]]
                    if sim_num > 7:
                        # 8,9
                        taus = [[np.arange(0.01,0.31,0.01)],
                                [np.arange(0.01,0.31,0.01)],
                                [np.arange(0.01,0.31,0.01)]]
                    tmp_gr, tmp_gr1 = [], []
                    for T in range(t_0+1,T_max + 1):
                        Is = [[b for idx_t,b in enumerate(a) if t[idx_sim][idx_t] > t_0 and t[idx_sim][idx_t] < T] for idx_sim, a in enumerate(I)]
                        growth_rates, growth_rates1 = [], []
                        for vec in Is:
                            # ( log(I(t+T)) - log(I(t)) ) / T
                            if len(vec) > 1:
                                growth_rates.append((np.log(vec[-1]) - np.log(vec[0])) / T)
                                # growth_rates.append((vec[-1] - vec[0]) / T)
                            else:
                                growth_rates.append(0)
                                # growth_rates1.append(0)
                        
                        tmp_gr.append(growth_rates)
                        # tmp_gr1.append(growth_rates1)
                    
                        # print(growth_rates, Is, taus[i][j])
                        # ax0.scatter([taus[i][j][tau]], [np.mean(growth_rates)], label = f'{T-t_0}' if sim_num == 0 and tau == 0 else '')
                        # ax1.scatter([taus[i][j][tau]], [r0], color='tab:orange', label = f'{model_names[j]}' if sim_num == 0 and tau == 0 else '')
                        
                        gr[-1][f'{T}'] = []
                        for row in tmp_gr:
                            for val in row:
                                gr[-1][f'{T}'].append(val)
                        # gr1[-1][f'{T}'] = []
                        # for row in tmp_gr1:
                        #     for val in row:
                        #         gr1[-1][f'{T}'].append(val)
                                
            except:
                print(f'no file {data} {model} {sim_num}')
                    
        ############################### fig 2 do r0 on x axis and growth rate y axis, different lines for multiple lines for each time frame ###########################
        for index , T in enumerate(range(t_0+1,T_max + 1)):
            x = bins
            y = [[] for _ in range(len(x))]
            y1 = [[] for _ in range(len(x))]
            # ax2.scatter(r0s, [np.mean([a[f'{T}']]) for a in gr], color = available_colors[index], label=f'{index}')
            for index1, g in enumerate(gr):    
                bin_idx = np.digitize(r0s[index1], digit, right=False)
                if bin_idx < len(y):
                    for val in g[f'{T}']:
                        y[bin_idx].append(val)
            
            # print(y)
            print('y\n\n\n')
            xs.append(x)
            ys.append([np.mean(a) for a in y])
            bot.append([np.percentile(a,5) if len(a) > 1 else 0 for a in y])
            top.append([np.percentile(a,95) if len(a) > 1 else 0 for a in y])
            print(index)
            # print(ys)
            
    np.savetxt(f'../output_data/simulations/big/sellke/growth_rate/figures/xs_{data}_{model}.csv', xs)
    np.savetxt(f'../output_data/simulations/big/sellke/growth_rate/figures/ys_{data}_{model}.csv',ys)
    np.savetxt(f'../output_data/simulations/big/sellke/growth_rate/figures/bot_{data}_{model}.csv',bot)
    np.savetxt(f'../output_data/simulations/big/sellke/growth_rate/figures/top_{data}_{model}.csv',top)