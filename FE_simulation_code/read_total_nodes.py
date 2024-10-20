import numpy as np
from odbAccess import *
import sys
import os



numbers = 10000

for i in range(0, numbers):

	odb = openOdb(path='./FE_simulation/simulation_odb/Job-' + str(i) + '.odb')

	assembly=odb.rootAssembly
	# assembly = mdb.models['Model-1'].rootAssembly
	part = assembly.instances['PART-21-1']

	# numNodes = 0

	coordinates_original = []
	for name, instance in assembly.instances.items():
		# n = len(part.nodes)
		# numNodes = numNodes + n
		with open(r'./FE_simulation_data/output_total_data/coordinates_original' + str(i) + '.txt', 'w') as info:
			for node in part.nodes:
				info.write(str(node.label) + str(node.coordinates) + '\n')
				coordinates_original.append(node.coordinates)
	np.savetxt('./FE_simulation_data/output_total_data/undeformed_data' + str(i) + '.txt', coordinates_original, fmt='%.07f')

	step1 = odb.steps['Step-1']
	lastFrame = step1.frames[-1]
	displacement_last = lastFrame.fieldOutputs['U']
	displacementValues_last = displacement_last.values

	DISP = []
	for v in displacementValues_last:
		DISP.append(v.data)

	np.savetxt('./FE_simulation_data/output_total_data/DISP' + str(i) + '.txt',DISP, fmt='%.07f')

	temp_coordinates_original = np.array(coordinates_original)
	temp_DISP = np.array(DISP)
	coordinates_deformed = temp_coordinates_original + temp_DISP


	with open(r'./FE_simulation_data/output_total_data/coordinates_deformed' + str(i) + '.txt', 'w') as info1:
		for ii, vv in enumerate(coordinates_deformed, start=1):
			info1.write(str(ii) + str(vv) + '\n')


	np.savetxt('./FE_simulation_data/output_total_data/deformed_data' + str(i) + '.txt', coordinates_deformed, fmt='%.07f')

	odb.close




