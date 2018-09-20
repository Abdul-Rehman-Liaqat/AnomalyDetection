

from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense, Reshape, LSTM, RepeatVector


def autoencoderNn(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Dense(7, input_shape=input_shape, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model



def autoencoderCnn(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=10, input_shape=input_shape, activation='relu'))
    model.add(Reshape(target_shape=(5,1)))
    model.add(Conv1D(filters=input_shape[0], kernel_size=5, activation='relu'))
    model.add(Reshape(target_shape=(input_shape[0],1)))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model

def autoencoderLstm(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(2, input_shape=input_shape))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(1, return_sequences=True))
    print(model.summary())
    model.compile(loss=loss, optimizer=optimizer)
    return model


def predictionCnn(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Conv1D(kernel_size=3, input_shape=input_shape, activation="relu", filters=5))
    model.add(Conv1D(kernel_size=2, input_shape=input_shape, activation="relu", filters=5))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model

def predictionLstm(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(20, input_shape = input_shape,activation = 'relu'))
    model.add(Dense(5,activation = 'relu'))
    model.add(Dense(1))
    print(model.summary())
    model.compile(loss=loss, optimizer=optimizer)
    return model

def predictionNn(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Dense(50,input_shape = input_shape, activation='relu'))
    model.add(Dense(25,input_shape = input_shape, activation='relu'))
    model.add(Dense(10,input_shape = input_shape, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model