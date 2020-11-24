import keras
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from numpy import savetxt

df = pd.read_csv('final_data.csv')

mid_prices = df['mid_price'].values
label = df['label'].values
# data = df[['mid_price', 'label']].values

seq_len = 70  # window size 지정
sequence_length = seq_len + 1

mid_price_result = []
for index in range(len(mid_prices) - sequence_length):
    mid_price_result.append(mid_prices[index: index + sequence_length])

label_result = []
for index in range(len(mid_prices) - sequence_length):
    label_result.append(mid_prices[index: index + sequence_length])

def normalize_windows(data):
    normalized_data = []
    for window in data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return np.array(normalized_data)

mid_price_result = normalize_windows(mid_price_result)
label_result = normalize_windows(label_result)

chart_result = mid_price_result

added_result = np.c_[mid_price_result, label_result]

#print(added_result.shape)
#savetxt('result_data.csv', result, delimiter=',')

added_result = pd.read_csv('result_data.csv')
added_result = added_result.values

#------------------------------------#

# split train and test data
row = int(len(added_result) * 0.67)
train = added_result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
y_train = train[:, -1]

x_test = added_result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
y_test = added_result[row:, -1]

print(x_train.shape, x_test.shape)

#------------------------------------#

# split train and test data of chart data
c_row = int(len(chart_result) * 0.67)
c_train = chart_result[:c_row, :]

cx_train = c_train[:, :-1]
cx_train = np.reshape(cx_train, (cx_train.shape[0], cx_train.shape[1], 1))
cy_train = c_train[:, -1]

cx_test = chart_result[c_row:, :-1]
cx_test = np.reshape(cx_test, (cx_test.shape[0], cx_test.shape[1], 1))
cy_test = chart_result[c_row:, -1]

print(cx_train.shape, cx_test.shape)

checkpoint_path = "training1.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

#------------------------------------#

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))   # seq_len = 70
model.add(keras.layers.Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=30,
          epochs=300,
          callbacks=[
              ModelCheckpoint(filepath=checkpoint_path,
                              monitor='val_loss',
                              verbose=1,
                              save_weights_only=True,
                              mode='auto'),

              ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=5,
                                verbose=1,
                                mode='auto')
          ])

pred = model.predict(x_test)
pred = pred.reshape(-1, 1)

#------------------------------------#

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(seq_len, 1)))   # seq_len = 70
model.add(keras.layers.Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(keras.layers.Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

model.fit(cx_train, cy_train,
          validation_data=(cx_test, cy_test),
          batch_size=30,
          epochs=300,
          callbacks=[
              ModelCheckpoint(filepath=checkpoint_path,
                              monitor='val_loss',
                              verbose=1,
                              save_weights_only=True,
                              mode='auto'),

              ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=5,
                                verbose=1,
                                mode='auto')
          ])

c_pred = model.predict(cx_test)

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='Real Value')
ax.plot(c_pred, label='Predicted with chart data')
ax.plot(pred, label='Predicted with chart data and text data')
ax.legend()
plt.show()
fig.savefig('Chart.png')