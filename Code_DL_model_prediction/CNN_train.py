import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import random
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_and_process_data(file_path):
    data = pd.read_csv(file_path, header=None, skiprows=1)
    inputs = []
    outputs = []

    for i in range(1, len(data), 2):
        input_up = data.iloc[i - 1, :10].to_numpy().astype(float)
        input_down = data.iloc[i, :10].to_numpy().astype(float)
        output_up = data.iloc[i - 1, 10:].to_numpy().astype(float) / float(1000)   ### Data scaling, the unit changed from mm to m
        output_down = data.iloc[i, 10:].to_numpy().astype(float) / float(1000)    ### Data scaling, the unit changed from mm to m
        input = np.array([input_up, input_down])
        output = np.array([output_up, output_down])
        inputs.append(input.reshape(2, 10))
        outputs.append(output.reshape(2, 200))

    return np.array(inputs), np.array(outputs)

#### Data processing of deformation curves
def Curve_unification(x):
    len_x = len(x)
    x1 = np.reshape(x, (len_x, 2, 200))
    x1_start = np.concatenate([np.full((len_x, 1, 200), x1[0, 0, 0]), np.full((len_x, 1, 200), x1[0, 1, 0])], axis=1)
    x1 = x1 - x1_start

    new_x1 = np.reshape(x1, (len_x, 2, 200))
    new_x1[:, [0, 1], :] = new_x1[:, [1, 0], :]
    new_x1[:, 0, :] = - new_x1[:, 0, :]

    return new_x1


train_file_path = './train_data.csv'
test_file_path = './test_data.csv'

train_inputs, train_outputs = load_and_process_data(train_file_path)
test_inputs, test_outputs = load_and_process_data(test_file_path)

train_outputs = Curve_unification(train_outputs)
test_outputs = Curve_unification(test_outputs)

def predicting(x):
    with tf.variable_scope("predicting", reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv1d(inputs=x, filters=320, kernel_size=3, padding='same')
        # conv1 = tf.layers.batch_normalization(conv1, training = True)
        conv1 = tf.nn.leaky_relu(conv1)
        conv2 = tf.layers.conv1d(inputs=conv1, filters=320, kernel_size=3, padding='same')
        conv2 = tf.nn.leaky_relu(conv2)
        conv3 = tf.layers.conv1d(inputs=conv2, filters=320, kernel_size=3, padding='same')
        conv3 = tf.nn.leaky_relu(conv3)
        conv4 = tf.layers.conv1d(inputs=conv3, filters=200, kernel_size=3, padding='same')

        return conv4

#### Data processing of deformation curves
def Curve_unification(x):
    x1 = np.reshape(x, (2, 200))
    x1_start = np.vstack([np.full((1, 200), x1[0, 0]), np.full((1, 200), x1[1, 0])])
    x1 = x1 - x1_start
    x1[0, :] = - x1[0, :]

    new_x1 = np.reshape(x1, (1, 2, 200))

    return new_x1


learning_rate = 1e-4

x = tf.placeholder(tf.float32, shape=[None, 2, 10])
y = tf.placeholder(tf.float32, shape=[None, 2, 200])

output_predict = predicting(x)


MSE_loss = tf.losses.mean_squared_error(output_predict, y)


Pre_vars = [v for v in tf.trainable_variables() if v.name.startswith('predicting')]

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
Pre_solver = optimizer.minimize(MSE_loss, var_list=Pre_vars)


init = tf.global_variables_initializer()
train_number_steps = 0
number_iterations = 161

batch_size = 60
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()
load_pretrain_model = True

with tf.Session(config=config) as sess:

    start_idx = 0
    print('Start init')
    sess.run(tf.global_variables_initializer())
    trainable_vars = tf.trainable_variables()
    total_weights = total_biases = 0

    for var in trainable_vars:
        shape = var.get_shape().as_list()
        total_elements = np.prod(shape)
        if 'bias' in var.name:
            total_biases += total_elements
        else:
            total_weights += total_elements

    print("Total weights:", total_weights)
    print("Total biases:", total_biases)

    iteration_times = []
    for iteration in range(number_iterations):
        start_time = time.time()
        print("iteration is ", iteration)
        train_number_steps += 100
        '''=========== Training the  Net  ======== '''


        for i in range(train_number_steps):
            indices = np.random.choice(train_inputs.shape[0], batch_size, replace=False)
            batch_inputs = train_inputs[indices]
            batch_outputs = train_outputs[indices]

            _, loss_predicting_data = sess.run([Pre_solver, MSE_loss],
                                               feed_dict={x: batch_inputs,
                                                          y: batch_outputs})

            print(iteration, i, '------loss_predicting_data------', loss_predicting_data)

        # saver.save(sess,
        #            './model_weights_CNN/model_weights_iteration' + str(iteration) + '.ckpt')

        '''=========== Testing the  Net  ======== '''
        test_number_steps = 2000
        for ii in range(test_number_steps):

            total_loss_test = sess.run(
                [MSE_loss],
                feed_dict={x: test_inputs, y: test_outputs})

            if ii % 50 == 0:
                test_curve = test_outputs[ii]

                predicting_curve = predicting_data[ii]
                test_output_data = np.reshape(test_curve, (2, 200))
                predicting_data = np.reshape(predicting_curve, (2, 200))
                plt.plot(test_output_data[0], test_output_data[1], marker='o', linestyle='-', markersize=3,
                         label='Target deformation')
                plt.plot(predicting_data[0], predicting_data[1], marker='o', linestyle='-', markersize=3,
                         label='Predict deformation')
                plt.xlabel('X_data', fontsize=12)
                plt.ylabel('Y_data', fontsize=12)
                plt.title('Deformed Curve', fontsize=14)
                plt.grid(True)
                plt.axis('equal')
                plt.legend()
                # plt.show()
                # print('test_output_data', test_output_data)
                # print('predicting_dat', predicting_data)
                plt.savefig('./WMT/CNN_predicting_fig/pre_fig_iteration' + str(iteration)
                            + '_test' + '_step' + str(ii) + '.png', dpi=300, bbox_inches='tight')
                plt.close()

        end_time = time.time()
        iteration_time = end_time - start_time
        print(iteration_time)
        iteration_times.append(iteration_time)

    iteration_times = np.array(iteration_times)
    sum_time = np.sum(iteration_times)
    total_train_time = np.append(iteration_times, [sum_time])
    df = pd.DataFrame(data=total_train_time)
    df.to_csv('./CNN_train_time/CNN_train_time.csv', index=False, header=False)

