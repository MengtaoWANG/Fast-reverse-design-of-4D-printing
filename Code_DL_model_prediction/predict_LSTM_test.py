import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector, LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=np.inf)
from keras.callbacks import CSVLogger
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
        output_up = data.iloc[i - 1, 10:].to_numpy().astype(float) / float(1000)  ### Data scaling, the unit is changed from mm to m
        output_down = data.iloc[i, 10:].to_numpy().astype(float) / float(1000)   ### Data scaling, the unit is changed from mm to m
        input = np.array([input_up, input_down])
        output = np.array([output_up, output_down])
        inputs.append(input.reshape(2, 10))
        outputs.append(output.reshape(2, 200))

    return np.array(inputs), np.array(outputs)

#### Data processing of deformation curves
def Curve_unification(x):
    len_x = len(x)
    # long_x = len(x[-2])
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
train_inputs = np.transpose(train_inputs, (0, 2, 1))
test_inputs = np.transpose(test_inputs, (0, 2, 1))

train_outputs = Curve_unification(train_outputs)
train_outputs = np.reshape(train_outputs, (6000, 2, 10, 20))
train_outputs = np.swapaxes(train_outputs, 1, 2)
train_outputs = np.reshape(train_outputs, (6000, 10, 40))

test_outputs = Curve_unification(test_outputs)
test_outputs = np.reshape(test_outputs, (2000, 2, 10, 20))
test_outputs = np.swapaxes(test_outputs, 1, 2)
test_outputs = np.reshape(test_outputs, (2000, 10, 40))

model = Sequential([
    LSTM(64, input_shape=(None, 2), return_sequences=True),
    LeakyReLU(alpha=0.1),
    LSTM(64, return_sequences=True),
    LeakyReLU(alpha=0.1),
    LSTM(64, return_sequences=True),
    LeakyReLU(alpha=0.1),
    TimeDistributed(Dense(40))
])

start_time = time.time()

loaded_model = load_model('./model_weights_LSTM/LSTM_model_weights_iteration80.h5')

for i in range(0, 2000):
    test_input = test_inputs[i]
    test_input = np.reshape(test_input, (1, 10, 2))
    predict_data = loaded_model.predict(test_input)
    predicting_data = np.reshape(predict_data, (10, 2, 20))
    predicting_data = np.swapaxes(predicting_data, 0, 1)
    predicting_data = np.reshape(predicting_data, (2, 200)).T
    df = pd.DataFrame(predicting_data)
    df.to_csv('./LSTM_test_outputs/output' + str(i) + '.csv', index=False)

end_time = time.time()

total_time = end_time - start_time
print('total time', total_time)





