import numpy as np
import pandas as pd
import csv
import os

input_combined_data = pd.DataFrame()

folder_path1 = './FE_simulation_data/input_data'
folder_path2 = './FE_simulation_data/output_data'

#input_data
file_count1 = 0

for filename in os.listdir(folder_path1):
    if file_count1 >= 10000:
        break
    if filename.endswith('.csv'):

        file_path = os.path.join(folder_path1, filename)


        data = pd.read_csv(file_path, header=None)

        data.columns = [f'input{i}' for i in range(1, len(data.columns) + 1)]

        input_combined_data = input_combined_data.append(data, ignore_index=True)
        file_count1 += 1

#output_data
output_combined_data = pd.DataFrame()

file_count2 = 0

for filename in os.listdir(folder_path2):

    if file_count2 >= 10000:
        break
    if filename.endswith('.csv'):

        file_path = os.path.join(folder_path2, filename)

        data = pd.read_csv(file_path, header=None)

        data.columns = [f'output{i}' for i in range(1, len(data.columns) + 1)]

        output_combined_data = output_combined_data.append(data, ignore_index=True)
        file_count2 += 1

train_data = pd.concat([input_combined_data, output_combined_data], axis=1)

train_data.to_csv('./FE_simulation_data/datasets/datasets.csv', index=False)
