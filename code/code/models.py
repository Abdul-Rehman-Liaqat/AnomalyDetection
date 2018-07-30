from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense, Reshape


def autoencoder_fully_connected(window_size,nb_features,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Dense(5,input_dim = window_size, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model

def autoencoder_fully_convolution(window_size,nb_features,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=10, input_shape=(window_size, nb_features), activation='relu'))
    model.add(Reshape(target_shape=(5,1)))
    model.add(Conv1D(filters=window_size, kernel_size=5, activation='relu'))
    model.add(Reshape(target_shape=(window_size,1)))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model


def autoencoder_lstm(window_size,nb_features,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(2, input_shape=(window_size, 1)))
    model.add(RepeatVector(window_size))
    model.add(LSTM(1, return_sequences=True))
    print(model.summary())
    model.compile(loss=loss, optimizer=optimizer)
    return model


def prediction_cnn(window_size,nb_features,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Conv1D(nb_filter=5, kernel_size=10, input_shape=(window_size, nb_features), activation='relu'))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model

def prediction_lstm(window_size,nb_features,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'relu'))
    print(model.summary())
    model.compile(loss=loss, optimizer=optimizer)
    return model

def prediction_nn(window_size,nb_features,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Dense(5,input_dim = window_size, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model