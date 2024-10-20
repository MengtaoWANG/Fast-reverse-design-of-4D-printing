import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import random
import json
import time
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import os
import csv
import itertools

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# np.set_printoptions(threshold=np.inf)


def load_and_process_data(file_path):
    data = pd.read_csv(file_path, header=None, skiprows=1)
    inputs = []
    outputs = []

    for i in range(1, len(data), 2):
        input_up = data.iloc[i - 1, :10].to_numpy().astype(float)
        input_down = data.iloc[i, :10].to_numpy().astype(float)
        output_up = data.iloc[i - 1, 10:].to_numpy().astype(float) / float(1000)
        output_down = data.iloc[i, 10:].to_numpy().astype(float) / float(1000)
        input = np.array([input_up, input_down])
        output = np.array([output_up, output_down])
        inputs.append(input.reshape(2, 10))
        outputs.append(output.reshape(2, 200))

    return np.array(inputs), np.array(outputs)

def Curve_unification(x):
    len_x = len(x)
    x1 = np.reshape(x, (len_x, 2, 200))
    x1_start = np.concatenate([np.full((len_x, 1, 200), x1[0, 0, 0]), np.full((len_x, 1, 200), x1[0, 1, 0])], axis=1)
    x1 = x1 - x1_start

    new_x1 = np.reshape(x1, (len_x, 2, 200))
    new_x1[:, [0, 1], :] = new_x1[:, [1, 0], :]
    new_x1[:, 0, :] = - new_x1[:, 0, :]

    return new_x1


test_file_path = './test_data/test_data.csv'

test_inputs, test_outputs = load_and_process_data(test_file_path)
test_outputs = Curve_unification(test_outputs)

def initialization_population(population_size):
    population = np.random.randint(2, size=(population_size, 1, 2, 10))
    return population
def fitness_function2(x, y):
    fit2 = - np.mean(np.square(x[0, :] - y[0, :]) + np.square(x[1, :] - y[1, :]))
    return fit2


def cross_mutate(x, number):
    x_elite = x[0: int(number * 0.05)]

    x_cross = []
    for _ in range(int(number * 0.76 / 2)):
        x_id1, x_id2 = np.random.choice(int(number), 2, replace=False)
        x_aa = x[x_id1].copy()
        x_bb = x[x_id2].copy()
        x_aa[:, 0, :5], x_aa[:, 1, 5:], x_bb[:, 1, :5], x_bb[:, 0, 5:] = x_bb[:, 0, :5].copy(), x_bb[:, 1, 5:].copy(), x_aa[:, 1, :5].copy(), x_aa[:, 0, 5:].copy()
        x_cross.append(x_aa)
        x_cross.append(x_bb)
    new_cross = np.array(x_cross)

    indices = np.random.choice(x.shape[0], 380, replace=False)
    x_mutate = x[indices]
    mutation_flag = np.random.rand(int(number * 0.19)) < 1
    for idx in np.where(mutation_flag)[0]:
        row, col = np.random.randint(0, 2), np.random.randint(0, 10)

        x_mutate[idx, 0, row, col] = 1 - x_mutate[idx, 0, row, col]

    new_populations = np.concatenate((x_elite, new_cross, x_mutate), axis=0)

    return new_populations


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

    times_per_iteration = []
    total_accuracy = []
    for ii in range(2000):
        # start_time = time.time()
        print('$$$$$$$$$$$$$$$$-------frequency------$$$$$$$$$$$$$---------------------------------', str(ii))
        target_input_data = test_inputs[ii]
        target_output_data = test_outputs[ii]
        print('target_input_data', target_input_data)
        skip = False
        total_input = initialization_population(population_size=populations_size)

        for gen in range(100):
            print('====================================generation===================================:' + str(gen))
            fit2_scores_list = []
            numbers = len(total_input)
            predicting_total_data = []

            for i in range(numbers):
                input_data = total_input[i, :, :, :]
                # input_data = np.array([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]])
                # print('input_data' + str(i), input_data)
                predicting_data = sess.run(output_predict,
                                                   feed_dict={x: input_data})

                predicting_data = np.reshape(predicting_data, (2, 200))
                predicting_total_data.append(predicting_data)

                fit2_scores = fitness_function2(predicting_data, target_output_data)
                fit2_scores_list.append(fit2_scores)

            if skip:
                break

            fit2_total_scores = np.array([fit2_scores_list])
            fit2_total_scores = np.reshape(fit2_total_scores, (numbers))

            highest_fit2_score = np.max(fit2_total_scores)
            # print("--------Highest fit2 score--------:", highest_fit2_score)
            highest_values_index = np.argmax(fit2_total_scores)
            # print('--------highest_values_index------', highest_values_index)
            optimal_individual = total_input[highest_values_index]
            # print('--------optimal_individual--------', optimal_individual)
            predicting_total_data = np.array(predicting_total_data)
            optimal_prediction = predicting_total_data[highest_values_index]


            sorted_indices = np.argsort(fit2_total_scores)[::-1]
            # print(sorted_indices)
            excellent_populations = total_input[sorted_indices]
            # print(excellent_populations)

            length = len(excellent_populations)
            excellent_populations = np.reshape(excellent_populations, (length, 1, 2, 10))
            crossed_mutated_populations = cross_mutate(excellent_populations, populations_size)
            total_input = crossed_mutated_populations
        # end_time = time.time()
        # interation_time = end_time - start_time
        # # print("Interation time: ", interation_time)
        # times_per_iteration.append(interation_time)
        optimal_individual = np.reshape(optimal_individual, (2, 10))
        print('optimal_individual', optimal_individual)
        comparison = np.all(target_input_data == optimal_individual, axis=0)
        overall_comparison = np.array_equal(target_input_data, optimal_individual)
        extended_comparison = np.append(comparison, overall_comparison).astype(int)
        # print('comparison', extended_comparison)
        total_accuracy.append(extended_comparison)
        if skip:
            continue
    total_accuracy1 = np.array(total_accuracy)
    accuracy = np.sum(total_accuracy1, axis=0) / 2000.
    print('accuracy', accuracy)
    df = pd.DataFrame(accuracy)
    df.to_csv('./TEA_reverse_design_accuracy.csv', index=False, header=False,
              encoding='utf-8')








