from utility import customLoss
from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense, Reshape, LSTM, RepeatVector, Conv2DTranspose




alpha = 0.2
previousLoss = [12,13,12,13,11,10]
y_true = [11,10.4,10.1,9.7,9.5]
y_pred = [9.9,10.0,10.5,10,9.8]

window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)



model = Sequential()
model.add(Dense(7, input_shape=input_shape, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(10, activation='relu'))
model.summary()
model.compile(loss=loss, optimizer=optimizer)

loss = customLoss(alpha,previousLoss)