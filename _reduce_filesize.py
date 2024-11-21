import os
import json

# Specify the directory you want to loop through
directory = '../output_data/simulations/big/sellke/'  # Replace with your directory path

# Loop through all files and directories in the specified directory
for filename in os.listdir(directory):
    # Construct full file path
    filepath = os.path.join(directory, filename)
    print(filepath)
    with open(filepath,'r') as f:
        result = json.load(f)
    reduced = {'SIR': result['SIR'], 't': result['t']}
    print(filepath[:-5] + '_red.json')
    with open(filepath[:-5] + '_red.json','w') as f:
        json.dump(reduced,f)