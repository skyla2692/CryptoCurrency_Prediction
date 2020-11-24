import keras
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

df = pd.read_csv('final_data.csv')

mid_prices = df['mid_price'].values
label = df['label'].values
# data = df[['mid_price', 'label']].values

seq_len = 70  # window size 지정
sequence_length = seq_len + 1
result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index: index + sequence_length])


def normalize_windows(data):
    normalized_data = []
    for window in data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return np.array(normalized_data)


result = normalize_windows(result)
print(result.shape)

# split train and test data
row = int(len(result) * 0.67)
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

print(x_train.shape, x_test.shape)

checkpoint_path = "training1.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(seq_len, 2)))   # seq_len = 70
model.add(keras.layers.Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(keras.layers.Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=30,
          epochs=100,
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

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()

# 가중치 확인
new_model = Sequential()
model.load_weights("training1.h5")

# 모델을 재평가합니다
loss, acc = model.evaluate(x_train, y_train, verbose=2)
print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))
