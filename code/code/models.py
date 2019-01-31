

from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense, Reshape, LSTM, RepeatVector, Conv2DTranspose


def autoencoderNn(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Dense(7, input_shape=input_shape, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(10))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model


def autoencoderNnAdaptive(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    if(input_shape[0] == 10):
        model.add(Dense(7, input_shape=input_shape, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(10))
    elif(input_shape[0] == 15):
        model.add(Dense(10, input_shape=input_shape, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(15))
    elif(input_shape[0] == 20):
        model.add(Dense(15, input_shape=input_shape, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(20))
    elif(input_shape[0] == 25):
        model.add(Dense(20, input_shape=input_shape, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(25))
    elif(input_shape[0] == 30):
        model.add(Dense(25, input_shape=input_shape, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(30))
    elif(input_shape[0] == 35):
        model.add(Dense(30, input_shape=input_shape, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(35))
    elif(input_shape[0] == 40):
        model.add(Dense(35, input_shape=input_shape, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(35, activation='relu'))
        model.add(Dense(40))
    elif(input_shape[0] == 45):
        model.add(Dense(40, input_shape=input_shape, activation='relu'))
        model.add(Dense(35, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(7, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(35, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(45))
    elif(input_shape[0] == 50):
        model.add(Dense(45, input_shape=input_shape, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(45, activation='relu'))
        model.add(Dense(50))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model

def autoencoderCnn(input_shape,loss='mse',optimizer='adam'):
#    model = Sequential()
#    model.add(Conv1D(filters=5, kernel_size=10, input_shape=input_shape, activation='relu'))
#    model.add(Reshape(target_shape=(5,1)))
#    model.add(Conv1D(filters=input_shape[0], kernel_size=5, activation='relu'))
#    model.add(Reshape(target_shape=(input_shape[0],1)))
#    model.summary()
#    model.compile(loss=loss, optimizer=optimizer)
#    return model

    model = Sequential()
    model.add(Conv1D(kernel_size=3, filters=5, strides = 2, input_shape=input_shape, activation="relu"))
    model.add(Conv1D(kernel_size=2, filters=5, input_shape=input_shape, activation="relu"))
    model.add(Dense(10, activation='relu'))
#        model.add(Reshape(target_shape=(0, 3, 10)))
#    model.add(Flatten())
    model.add(Reshape((0,) + model.output_shape))
    print(model.output_shape)
    model.summary()
#    model.add(Dense(1))
#    model.add(Conv2DTranspose(kernel_size=2, filters=5, input_shape=input_shape, activation="relu"))
#    model.summary()
#    model.compile(loss=loss, optimizer=optimizer)
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
    model.add(Conv1D(kernel_size=3, filters=5, strides = 2, input_shape=input_shape, activation="relu"))
    model.add(Conv1D(kernel_size=2, filters=5, input_shape=input_shape, activation="relu"))
    model.add(Dense(10, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model

def predictionLstm(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(50, input_shape = input_shape,activation = 'relu'))
    model.add(Dense(5))
    model.add(Dense(1))
    print(model.summary())
    model.compile(loss=loss, optimizer=optimizer)
    return model

def predictionCnnStepAhead(input_shape,nStep,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Conv1D(kernel_size=4, filters=8, strides = 2, input_shape=input_shape, activation="relu"))
    model.add(Conv1D(kernel_size=4, filters=8, strides = 2, input_shape=input_shape, activation="relu"))
    model.add(Conv1D(kernel_size=3, filters=10, strides = 2, input_shape=input_shape, activation="relu"))
    model.add(Conv1D(kernel_size=3, filters=10, strides = 2, input_shape=input_shape, activation="relu"))
    model.add(Conv1D(kernel_size=2, filters=12, input_shape=input_shape, activation="relu"))
    model.add(Dense(10, activation='relu'))
    model.add(Flatten())
    model.add(Dense(nStep))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model

def predictionLstmStepAheadHuge(input_shape,nStep,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(300, input_shape = input_shape,activation = 'relu'))
    model.add(Dense(100))
    model.add(Dense(nStep))
    print(model.summary())
    model.compile(loss=loss, optimizer=optimizer)
    return model


def predictionLstmStepAhead(input_shape,nStep,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(50, input_shape = input_shape,activation = 'relu'))
    model.add(Dense(nStep))
#    model.add(Dense(1))
    print(model.summary())
    model.compile(loss=loss, optimizer=optimizer)
    return model

def predictionLstmStepAheadWithErrorFeed(input_shape,nStep,loss='mse',optimizer='adam', errorInput = 5):
    model = Sequential()
    model.add(LSTM(50, input_shape[0]+errorInput,input_shape[1],activation = 'relu'))
    model.add(Dense(nStep))
#    model.add(Dense(1))
    print(model.summary())
    model.compile(loss=loss, optimizer=optimizer)
    return model


def predictionNn(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Dense(50,input_shape = input_shape, activation='relu'))
    model.add(Dense(25,input_shape = input_shape, activation='relu'))
    model.add(Dense(10,input_shape = input_shape, activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model

def predictionNnStepAhead(input_shape,nStep,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Dense(50,input_shape = input_shape, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(nStep))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model

def predictionNnStepAheadWithErrorFeed(input_shape,nStep,loss='mse',optimizer='adam', errorInput = 5):
    model = Sequential()
    model.add(Dense(50,input_shape = (input_shape[0]+errorInput,), activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(nStep))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model


def predictionNnWithRecency(input_shape,loss='mse',optimizer='adam'):
    model = Sequential()
    model.add(Dense(50,input_shape = input_shape, activation='relu'))
    model.add(Dense(25,input_shape = input_shape, activation='relu'))
    model.add(Dense(10,input_shape = input_shape, activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    return model