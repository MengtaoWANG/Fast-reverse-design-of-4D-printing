import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import random
import json
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import os
import csv
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.set_printoptions(threshold=np.inf)


def initialization_population(population_size):
    population = np.random.randint(2, size=(population_size, 1, 2, 10))
    return population

# def Curve_unification(x):
#     x1 = x
#     x1_start = np.vstack([np.full((1, 200), x1[0, 0]), np.full((1, 200), x1[1, 0])])
#     x1 = x1 - x1_start
#     x1_endpoint_dis = np.sqrt(np.square(x1[0, -1]) + np.square(x1[1, -1]))
#     vector_one = x1[:, -1]
#     vector_two = np.array([x1_endpoint_dis, 0.])
#     cos_angle = ((vector_one[0] * vector_two[0] + vector_one[1] * vector_two[1]) /
#              (np.sqrt(np.square(vector_one[0]) + np.square(vector_one[1])) * np.sqrt(np.square(vector_two[0]) + np.square(vector_two[1]))))
#     angle = np.arccos(cos_angle)
#     angle_matrix_ni = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
#     angle_matrix_shun = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
#     if x1[1, -1] < 0:
#         new_x1 = np.dot(angle_matrix_ni, x1)
#     else:
#         new_x1 = np.dot(angle_matrix_shun, x1)
#
#     return new_x1

def fitness_function1(x, y, gen):
    a = gen * 20
    b = (gen + 1) * 20 - 1
    fit1 = - np.mean(np.square(x[0, a:b] - y[0, a:b]) + np.square(x[1, a:b] - y[1, a:b]))
    return fit1

def fitness_function2(x, y):
    fit2 = - np.mean(np.square(x[0, :] - y[0, :]) + np.square(x[1, :] - y[1, :]))
    return fit2


def cross_function(x, number, gen):

    length_x = len(x)
    x1 = x[0: int(number/20)]

    if gen > 3:
        gen = 3

    x_cross = []
    for _ in range(int(19*number/40)):
        x_id1, x_id2 = np.random.choice(int(length_x - number/20), 2, replace=False)
        x_aa = x[x_id1].copy()
        x_bb = x[x_id2].copy()
        x_aa[:, 1, gen+1:], x_bb[:, 1, gen+1:] = x_bb[:, 1, gen+1:].copy(), x_aa[:, 1, gen+1:].copy()
        x_cross.append(x_aa)
        x_cross.append(x_bb)
    x_cross = np.array(x_cross)
    new_x = np.concatenate((x1, x_cross), axis=0)

    return new_x

def mutate_function(x, gen):
    if gen > 3:
        gen = 3
    sub_arrays = x[:, :, :, gen + 1]
    sub_array_types = [[0, 0], [0, 1], [1, 0], [1, 1]]
    counts = {str(arr_type): np.sum(np.all(sub_arrays == arr_type, axis=2)) for arr_type in sub_array_types}

    while any(count < 430 for count in counts.values()):
        min_count_type = min(counts, key=counts.get)
        max_count_type = max(counts, key=counts.get)
        num_needed = 430 - counts[min_count_type]

        indices = np.where(np.all(sub_arrays == np.array(eval(max_count_type)), axis=2))[0]
        selected_indices = np.sort(indices)[-num_needed:]

        for idx in selected_indices:
            x[idx, :, :, gen+1] = np.array(eval(min_count_type))
        counts = {str(arr_type): np.sum(np.all(sub_arrays == arr_type, axis=2)) for arr_type in sub_array_types}

        mutation_flag = np.random.rand(2000) < 0.1
        if gen <= 2:
            for idx in np.where(mutation_flag)[0]:
                row, col = np.random.randint(0, 2), np.random.randint(gen+2, 10)

                x[idx, 0, row, col] = 1 - x[idx, 0, row, col]

    return x

def all_possibilities(x, gen):
    combinations = list(itertools.product([0, 1], repeat=10))
    array_2_10 = np.array(combinations).reshape(-1, 2, 5)
    array_2_10 = np.reshape(array_2_10, (1024, 1, 2, 5))
    x_5 = x[0, :, :, :gen+1]
    x_5 = np.tile(x_5, (1024, 1, 1, 1))
    x_all = np.concatenate((x_5, array_2_10), axis=3)

    return x_all

def networks_1(x, filter):
    x_cut = tf.split(x, num_or_size_splits=10, axis=2)
    conv1_0 = tf.layers.conv1d(inputs=x_cut[0], filters=filter, kernel_size=3, padding='same')
    conv1_0 = tf.nn.leaky_relu(conv1_0)
    conv1_1 = tf.layers.conv1d(inputs=x_cut[1], filters=filter, kernel_size=3, padding='same')
    conv1_1 = tf.nn.leaky_relu(conv1_1)
    conv1_2 = tf.layers.conv1d(inputs=x_cut[2], filters=filter, kernel_size=3, padding='same')
    conv1_2 = tf.nn.leaky_relu(conv1_2)
    conv1_3 = tf.layers.conv1d(inputs=x_cut[3], filters=filter, kernel_size=3, padding='same')
    conv1_3 = tf.nn.leaky_relu(conv1_3)
    conv1_4 = tf.layers.conv1d(inputs=x_cut[4], filters=filter, kernel_size=3, padding='same')
    conv1_4 = tf.nn.leaky_relu(conv1_4)
    conv1_5 = tf.layers.conv1d(inputs=x_cut[5], filters=filter, kernel_size=3, padding='same')
    conv1_5 = tf.nn.leaky_relu(conv1_5)
    conv1_6 = tf.layers.conv1d(inputs=x_cut[6], filters=filter, kernel_size=3, padding='same')
    conv1_6 = tf.nn.leaky_relu(conv1_6)
    conv1_7 = tf.layers.conv1d(inputs=x_cut[7], filters=filter, kernel_size=3, padding='same')
    conv1_7 = tf.nn.leaky_relu(conv1_7)
    conv1_8 = tf.layers.conv1d(inputs=x_cut[8], filters=filter, kernel_size=3, padding='same')
    conv1_8 = tf.nn.leaky_relu(conv1_8)
    conv1_9 = tf.layers.conv1d(inputs=x_cut[9], filters=filter, kernel_size=3, padding='same')
    conv1_9 = tf.nn.leaky_relu(conv1_9)

    return conv1_0, conv1_1, conv1_2, conv1_3, conv1_4, conv1_5, conv1_6, conv1_7, conv1_8, conv1_9


def networks_2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, filter):
    conv2_0 = tf.layers.conv1d(inputs=x0, filters=filter, kernel_size=3, padding='same')
    conv2_0 = tf.nn.leaky_relu(conv2_0)
    conv2_1 = tf.layers.conv1d(inputs=conv2_0 + x1, filters=filter, kernel_size=3, padding='same')
    conv2_1 = tf.nn.leaky_relu(conv2_1)
    conv2_2 = tf.layers.conv1d(inputs=conv2_1 + x2, filters=filter, kernel_size=3, padding='same')
    conv2_2 = tf.nn.leaky_relu(conv2_2)
    conv2_3 = tf.layers.conv1d(inputs=conv2_2 + x3, filters=filter, kernel_size=3, padding='same')
    conv2_3 = tf.nn.leaky_relu(conv2_3)
    conv2_4 = tf.layers.conv1d(inputs=conv2_3 + x4, filters=filter, kernel_size=3, padding='same')
    conv2_4 = tf.nn.leaky_relu(conv2_4)
    conv2_5 = tf.layers.conv1d(inputs=conv2_4 + x5, filters=filter, kernel_size=3, padding='same')
    conv2_5 = tf.nn.leaky_relu(conv2_5)
    conv2_6 = tf.layers.conv1d(inputs=conv2_5 + x6, filters=filter, kernel_size=3, padding='same')
    conv2_6 = tf.nn.leaky_relu(conv2_6)
    conv2_7 = tf.layers.conv1d(inputs=conv2_6 + x7, filters=filter, kernel_size=3, padding='same')
    conv2_7 = tf.nn.leaky_relu(conv2_7)
    conv2_8 = tf.layers.conv1d(inputs=conv2_7 + x8, filters=filter, kernel_size=3, padding='same')
    conv2_8 = tf.nn.leaky_relu(conv2_8)
    conv2_9 = tf.layers.conv1d(inputs=conv2_8 + x9, filters=filter, kernel_size=3, padding='same')
    conv2_9 = tf.nn.leaky_relu(conv2_9)

    return conv2_0, conv2_1, conv2_2, conv2_3, conv2_4, conv2_5, conv2_6, conv2_7, conv2_8, conv2_9

def networks_3(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, filter):
    conv3_0 = tf.layers.conv1d(inputs=x0, filters=filter, kernel_size=3, padding='same')
    conv3_0 = tf.nn.leaky_relu(conv3_0)
    conv3_1 = tf.layers.conv1d(inputs=conv3_0 + x1, filters=filter, kernel_size=3, padding='same')
    conv3_1 = tf.nn.leaky_relu(conv3_1)
    conv3_2 = tf.layers.conv1d(inputs=conv3_1 + x2, filters=filter, kernel_size=3, padding='same')
    conv3_2 = tf.nn.leaky_relu(conv3_2)
    conv3_3 = tf.layers.conv1d(inputs=conv3_2 + x3, filters=filter, kernel_size=3, padding='same')
    conv3_3 = tf.nn.leaky_relu(conv3_3)
    conv3_4 = tf.layers.conv1d(inputs=conv3_3 + x4, filters=filter, kernel_size=3, padding='same')
    conv3_4 = tf.nn.leaky_relu(conv3_4)
    conv3_5 = tf.layers.conv1d(inputs=conv3_4 + x5, filters=filter, kernel_size=3, padding='same')
    conv3_5 = tf.nn.leaky_relu(conv3_5)
    conv3_6 = tf.layers.conv1d(inputs=conv3_5 + x6, filters=filter, kernel_size=3, padding='same')
    conv3_6 = tf.nn.leaky_relu(conv3_6)
    conv3_7 = tf.layers.conv1d(inputs=conv3_6 + x7, filters=filter, kernel_size=3, padding='same')
    conv3_7 = tf.nn.leaky_relu(conv3_7)
    conv3_8 = tf.layers.conv1d(inputs=conv3_7 + x8, filters=filter, kernel_size=3, padding='same')
    conv3_8 = tf.nn.leaky_relu(conv3_8)
    conv3_9 = tf.layers.conv1d(inputs=conv3_8 + x9, filters=filter, kernel_size=3, padding='same')
    conv3_9 = tf.nn.leaky_relu(conv3_9)

    conv3_output = tf.concat([conv3_0, conv3_1, conv3_2, conv3_3, conv3_4,
                              conv3_5, conv3_6, conv3_7, conv3_8, conv3_9], axis=-1)
    return conv3_output

def predicting(x):
    with tf.variable_scope("predicting", reuse=tf.AUTO_REUSE):
        net1_0, net1_1, net1_2, net1_3, net1_4, net1_5, net1_6, net1_7, net1_8, net1_9 = networks_1(x, 32)
        net2_0, net2_1, net2_2, net2_3, net2_4, net2_5, net2_6, net2_7, net2_8, net2_9 = networks_2(net1_0, net1_1,
                                                                                                    net1_2, net1_3,
                                                                                                    net1_4, net1_5,
                                                                                                    net1_6, net1_7,
                                                                                                    net1_8, net1_9,
                                                                                                    32)
        Net3_output = networks_3(net2_0, net2_1, net2_2, net2_3, net2_4, net2_5, net2_6, net2_7, net2_8, net2_9, 32)
        Net4_output = tf.layers.conv1d(inputs=Net3_output, filters=200, kernel_size=3, padding='same')
        return Net4_output

x = tf.placeholder(tf.float32, shape=[None, 2, 10])
y = tf.placeholder(tf.float32, shape=[None, 2, 200])

output_predict = predicting(x)


a = tf.Variable(0., trainable=False, dtype=tf.float32, name='a')
b = tf.Variable(0., trainable=False, dtype=tf.float32, name='b')

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()
load_pretrain_model = True
populations_size = 2000

'''=========== Testing the  Net  ======== '''
with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './model_weights_SEP_CNN/model_weights_iteration80.ckpt')

    target_output_data = pd.read_csv('./target_shape/target_test.csv', header=None)
    target_output_data = target_output_data.to_numpy()
    for ii in range(1000):
        print('$$$$$$$$$$$$$$$$-------frequency------$$$$$$$$$$$$$---------------------------------', str(ii))
        skip = False
        total_input = initialization_population(population_size=populations_size)
        chromosomes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        for gen in range(6):
            print('====================================generation===================================:' + str(gen))
            scores_dict = {str(array): [] for array in chromosomes}
            fit2_scores_list = []
            numbers = len(total_input)
            predicting_total_data = []

            for i in range(numbers):
                input_data = total_input[i, :, :, :]
                # input_data = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]])
                # print('input_data' + str(i), input_data)
                input_data_gen = np.reshape(input_data[:, :, gen], (2))
                predicting_data = sess.run(output_predict,
                                                   feed_dict={x: input_data})

                predicting_data = np.reshape(predicting_data, (2, 200))
                predicting_total_data.append(predicting_data)

                fit_score = fitness_function1(predicting_data, target_output_data, gen)
                fit2_scores = fitness_function2(predicting_data, target_output_data)
                fit2_scores_list.append(fit2_scores)
                for chromosome in chromosomes:
                    if np.array_equal(input_data_gen, chromosome):
                        scores_dict[str(chromosome)].append(fit_score)
                        # print(scores_dict)
                if fit2_scores >= -(1e-6):
                    print('--------generation:' + str(gen) + '-----' + 'number:' + str(i))
                    csv_file_path = './PEA_gen_overall_crossover.csv'
                    data_to_write = [gen, i]
                    with open(csv_file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(data_to_write)

                    print('=====input_data=====', input_data)
                    print("-------Highest score--------:", str(gen), fit2_scores)
                    plt.plot(predicting_data[0], predicting_data[1], marker='o', linestyle='-', markersize=3,
                             label='Prediction deformation')
                    plt.plot(target_output_data[0], target_output_data[1], marker='o', linestyle='-', markersize=3,
                             label='Target deformation')
                    plt.xlabel('X_data', fontsize=12)
                    plt.ylabel('Y_data', fontsize=12)
                    plt.title('Deformed Curve' + str(gen), fontsize=14)
                    plt.grid(False)
                    plt.axis('equal')
                    plt.legend()
                    # plt.show()

                    skip = True
                    break
            if skip:
                break

            average_scores = {array: np.mean(scores) if scores else 0 for array, scores in scores_dict.items()}
            # print(average_scores)
            fit2_total_scores = np.array([fit2_scores_list])
            fit2_total_scores = np.reshape(fit2_total_scores, (numbers, 1))

            highest_average_score = max(average_scores.values())
            highest_fit2_score = np.max(fit2_total_scores)
            # print("--------Highest fit2 score--------:", highest_fit2_score)
            highest_values_index = np.argmax(fit2_total_scores)
            # print('--------highest_values_index------', highest_values_index)
            optimal_individual = total_input[highest_values_index]
            # print('--------optimal_individual--------', optimal_individual)
            predicting_total_data = np.array(predicting_total_data)
            optimal_prediction = predicting_total_data[highest_values_index]

            # plt.plot(optimal_prediction[0], optimal_prediction[1], marker='o', linestyle='-', markersize=3,
            #          label='Prediction deformation')
            # plt.plot(target_output_data[0], target_output_data[1], marker='o', linestyle='-', markersize=3,
            #          label='Target deformation')
            # plt.xlabel('X_data', fontsize=12)
            # plt.ylabel('Y_data', fontsize=12)
            # plt.title('Deformed Curve' + str(gen), fontsize=14)
            # plt.grid(False)
            # plt.axis('equal')
            # plt.legend()
            # plt.show()

            highest_scoring_arrays = [array for array, score in average_scores.items() if
                                      score == highest_average_score]
            array_elements = highest_scoring_arrays[0].strip('[]').split()
            highest_array = np.array([int(element) for element in array_elements])
            print('---------optimal block---------', highest_array)
            # print("-------Highest average score--------:", str(gen), highest_average_score)

            mask = (total_input[:, :, :, gen] == highest_array).all(axis=2)
            filtered_array = total_input[mask]
            fit2_filtered_array = fit2_total_scores[mask]
            # print(fit2_filtered_array)
            sorted_indices = np.argsort(fit2_filtered_array)[::-1]
            # print(sorted_indices)
            excellent_populations = filtered_array[sorted_indices]
            # print(excellent_populations)

            length = len(excellent_populations)
            excellent_populations = np.reshape(excellent_populations, (length, 1, 2, 10))
            if gen < 4:
                Crossed_populations = cross_function(excellent_populations, populations_size, gen)
                Mutated_populations = mutate_function(Crossed_populations, gen)

                total_input = Mutated_populations
                # print('------total input---------', str(gen), total_input.shape)
            if gen == 4:
                all_populations = all_possibilities(excellent_populations, gen)
                total_input = all_populations
                # print('------total input---------', str(gen), total_input)


        if skip:
            continue







