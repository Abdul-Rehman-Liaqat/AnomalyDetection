from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense


def autoencoder_fully_connected(window_size,nb_features,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Dense(5,input_dim = window_size, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    return model