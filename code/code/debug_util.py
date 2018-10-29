

previous_loss = np.array([122,92,91,81,75,45,91])
y_true = np.array([0.12,0.15,0.17,0.15,0.16,0.18,0.20,0.18,0.15])
y_pred = np.array([0.14,0.13,0.18,0.12,0.16,0.13,0.16,0.18,0.2])


def mse(y_true,y_pred):
    res = y_true - y_pred
    return sum(res**2)


def customLoss(previous_loss):
    def lossFunction(y_true, y_pred):
        loss = mse(y_true, y_pred)
        exp_weights = create_exponential_weights(alpha,len(previous_loss))
        previous_loss = np.sum(np.multiply(exp_weights,previous_loss))
        loss += k.sum(previous_loss,loss*alpha)
        return loss
    return lossFunction



    #def customLoss(previous_loss, weight='equal'):
#    def lossFunction(y_true, y_pred):
        loss = mse(y_true, y_pred)
        loss += K.sum(val, K.abs(K.sum(K.square(layer_weights), axis=1)))
  #      return loss

 #   return lossFunction


import pandas as pd
from utility import customLoss
from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense, Reshape, LSTM, RepeatVector, Conv2DTranspose
import numpy as np


alpha = 0.6
L = 7

df = pd.read_csv('/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/select_data/yahoo/real_1_9_18_23_32_49_59.csv')
optimizer='adam'
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)

model = Sequential()
model.add(Dense(7, input_shape=input_shape, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(10, activation='relu'))
model.compile(loss=customLoss(previous_loss), optimizer=optimizer)
model.summary()

model.compile(loss=customLoss(weights,0.03), optimizer =..., metrics = ...)

    error_prediction = []
    L = []
    for i in np.arange(len(df) - input_shape[0]):
        X_input = max_min_normalize(df["value"].values[i:i+(input_shape[0])], max_min_var)
        X_input = X_input.reshape((1,)+input_shape)
        pred = model.predict(X_input)
        error_prediction.append(np.sqrt((pred-X_input)*(pred-X_input))[0][0])
        history = model.fit(X_input,X_input , nb_epoch=nb_epoch, verbose=0)
        L.append(score_postprocessing(error_prediction,len(error_prediction)))
    temp_no_error = [0]*(input_shape[0])
    error_prediction = temp_no_error + error_prediction
    L[0] = 0.5
    L_no_error = [0.5]*(input_shape[0])
    L = L_no_error + L
    df['error_prediction'] = error_prediction
    df['anomaly_score'] = L
    return df