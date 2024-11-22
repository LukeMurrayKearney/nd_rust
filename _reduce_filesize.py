# import os
# import json

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
#         reduced = {'SIR': result['SIR'], 't': result['t']}
#         print(filepath[:-5] + '_red.json')
#         with open(filepath[:-5] + '_red.json','w') as f:
#             json.dump(reduced,f)
#             f.close()
#     except Exception as e: 
#         print(f"Error processing file {filepath}: {e}")
#         continue



import os
import re

def delete_files_with_three_numbers(directory):
    """
    Deletes files in the specified directory that have exactly three numbers in a row in their filenames.

    Parameters:
    directory (str): The path to the directory to search in.

    Returns:
    None
    """
    # Regular expression pattern to match filenames with exactly three numbers in a row
    pattern = re.compile(r'.*[^0-9][0-9]{3}[^0-9].*|^[0-9]{3}[^0-9].*|.*[^0-9][0-9]{3}$|^[0-9]{3}$')

    # List all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Check if it's a file (not a directory)
        if os.path.isfile(filepath):
            # Check if the filename matches the pattern
            if pattern.match(filename):
                print(f"Deleting file: {filepath}")
                # Uncomment the following line to actually delete the files
                os.remove(filepath)
            else:
                print(f"Skipping file: {filepath}")
        else:
            print(f"Skipping non-file: {filepath}")

# Example usage:
# Replace '/path/to/your/directory' with the actual path
directory_path = '../output_data/simulations/big/sellke/'
delete_files_with_three_numbers(directory_path)