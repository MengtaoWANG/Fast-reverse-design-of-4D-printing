import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


with open('./FE_simulation_data/Nodes of 200 deformation coordinates.csv', 'r', encoding='utf-8-sig') as node_code:
    code_reader = csv.reader(node_code)
    code_list1 = []
    for row in code_reader:
        code_list1.extend(row)
    code_list = []
    for string in code_list1:
        integer = int(string)
        integer = integer - 1
        code_list.append(integer)

total_numbers = 10000

for ii in range(0, total_numbers):
    # print(code_list)
    YZ_code_data = []
    Y_DATA = []
    Z_DATA = []

    for i in code_list:
        with open('./FE_simulation_data/output_total_data/deformed_data'+ str(ii) + '.txt', 'r') as path1:
            lines = path1.readlines()
            XYZ_parts = lines[i].strip().split()

            Y_data = float(XYZ_parts[1])
            Y_DATA.append(Y_data)

            Z_data = float(XYZ_parts[2])
            Z_DATA.append(Z_data)

            YZ_code_data.append([Y_data, Z_data])

    YZ_DATA = [Y_DATA, Z_DATA]

    with open('./FE_simulation_data/output_data/output_data' + str(ii) +'.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        for row in YZ_DATA:
            csv_writer.writerow(row)

    # print(Y_DATA)
    # print(YZ_code_data)
            # XYZ_CODE_DATA = float(XYZ_code_data)
            # print(type(XYZ_CODE_DATA))
    # print(Y_DATA)
    # print(Z_DATA)
    # plt.figure(figsize=(8, 6))
    # plt.plot(Z_DATA, Y_DATA, marker='o', linestyle='-', markersize=3, label='Deformation Curve')
    # plt.xlabel('X_data', fontsize=12)
    # plt.ylabel('Y_data', fontsize=12)
    # plt.title('Deformed Data', fontsize=14)
    # plt.grid(True)
    # plt.axis('equal')
    # plt.legend()
    # # plt.show()
    # plt.savefig('./FE_simulation_data/deformation_figure/def_fig_' + str(ii) + '.png', dpi=300, bbox_inches='tight')
    # plt.close()


