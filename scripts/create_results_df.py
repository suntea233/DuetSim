import pandas as pd
import os

results_folder = 'results/477f455dc2964e34abe1c8ce2cdd8aa2'
output_file = 'results.csv'

results_files = os.listdir(results_folder)
results = []
for filename in os.listdir(results_folder):
    if filename == output_file:
        continue
    params = filename.split('|')[1:]
    params_dict = {}
    for param in params:
        key, val = param.split(':')
        params_dict[key] = val
    filepath = os.path.join(results_folder, filename, 'ours', 'res.txt')
    with open(filepath, 'r') as fr:
        lines = fr.readlines()
        if lines:
            for line in lines:
                key_val = line.split(':')
                if len(key_val) == 2:
                    key, val = key_val
                    key = key.strip()
                    val = val.strip()
                    params_dict[key] = val
            results.append(params_dict)

df = pd.DataFrame.from_records(results)
df.to_csv(os.path.join(results_folder, output_file), index=False)
