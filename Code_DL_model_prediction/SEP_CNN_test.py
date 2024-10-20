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


def ADSE_loss(x, y):
    channel_1_x = tf.square(x[:, 0, 1:] - x[:, 0, :-1])
    channel_2_x = tf.square(x[:, 1, 1:] - x[:, 1, :-1])
    result_x = channel_1_x + channel_2_x
    result_x = tf.sqrt(result_x)

    channel_1_y = tf.square(y[:, 0, 1:] - y[:, 0, :-1])
    channel_2_y = tf.square(y[:, 1, 1:] - y[:, 1, :-1])
    result_y = channel_1_y + channel_2_y
    result_y = tf.sqrt(result_y)

    loss = tf.reduce_sum(tf.reduce_mean(tf.square(result_x - result_y), axis=0))

    return loss


def ACSE_loss(x, y):
    x1 = x[:, 0:, 1:199] - x[:, 0:, 0:198]
    x2 = x[:, 0:, 2:200] - x[:, 0:, 1:199]
    total_angle_x = (x1[:, 0] * x2[:, 0] + x1[:, 1] * x2[:, 1]) / (tf.sqrt(tf.square(x1[:, 0]) + tf.square(x1[:, 1])) + tf.sqrt(tf.square(x2[:, 0]) + tf.square(x2[:, 1])))

    y1 = y[:, 0:, 1:199] - y[:, 0:, 0:198]
    y2 = y[:, 0:, 2:200] - y[:, 0:, 1:199]
    total_angle_y = (y1[:, 0] * y2[:, 0] + y1[:, 1] * y2[:, 1]) / (tf.sqrt(tf.square(y1[:, 0]) + tf.square(y1[:, 1])) + tf.sqrt(tf.square(y2[:, 0]) + tf.square(y2[:, 1])))

    loss = tf.reduce_sum(tf.reduce_mean(tf.square(total_angle_x - total_angle_y), axis=0))
    return loss



learning_rate = 1e-4

x = tf.placeholder(tf.float32, shape=[None, 2, 10])
y = tf.placeholder(tf.float32, shape=[None, 2, 200])


output_predict = predicting(x)

loss1 = tf.losses.mean_squared_error(output_predict, y)
loss2 = ADSE_loss(output_predict, y)
loss3 = ACSE_loss(output_predict, y)

a = tf.Variable(0., trainable=False, dtype=tf.float32, name='a')
b = tf.Variable(0., trainable=False, dtype=tf.float32, name='b')

total_loss = 1.0 * loss1 + a * loss2 + b * loss3


Pre_vars = [v for v in tf.trainable_variables() if v.name.startswith('predicting')]

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
Pre_solver = optimizer.minimize(total_loss, var_list=Pre_vars)


init = tf.global_variables_initializer()
train_number_steps = 0
number_iterations = 81  # in each iteration, the receiver, the transmitter and the channel will be updated

num_samples = train_inputs.shape[0]
batch_size = 60
number_batches = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()
load_pretrain_model = True

with tf.Session(config=config) as sess:
    start_idx = 0
    print('Start init')
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    saver.restore(sess, './model_weights_SEP_CNN/model_weights_iteration80.ckpt')

    for i in range(2000):
        test_input = test_inputs[i]
        test_input = np.reshape(test_input, (1, 2, 10))
        predicting_data = sess.run(output_predict,
                                   feed_dict={x: test_input})
        predict_data = np.reshape(predicting_data, (2, 200)).T
        df = pd.DataFrame(predict_data)
        df.to_csv('./SEP-CNN test outputs/output' + str(i) + '.csv', index=False)

    end_time = time.time()

    total_time = end_time - start_time
    print('total time', total_time)
