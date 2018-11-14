import numpy as np
from utility import *

previous_loss = np.array([122,92,91,81,75,45,91])
y_true = np.array([0.12,0.15,0.17,0.15,0.16,0.18,0.20,0.18,0.15])
y_pred = np.array([0.14,0.13,0.18,0.12,0.16,0.13,0.16,0.18,0.2])

def create_exponential_weights(alpha,L):
    exp_weights = np.array([])
    for i in range(L-1):
        exp_weights = np.append(exp_weights,(1-alpha)**(i+1))
    exp_weights = alpha*exp_weights
    exp_weights = np.append(exp_weights,(1-alpha)**L)
    return exp_weights

# def mse(y_true,y_pred):
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     diff = y_true - y_pred
#     return sum(diff**2)

def customLoss(alpha,previousLoss):
    def lossFunction(y_true, y_pred):
        loss = mse(y_true, y_pred)
        exp_weights = create_exponential_weights(alpha,len(previousLoss))
        previous_loss = np.sum(np.multiply(exp_weights,previousLoss))
#        loss = previous_loss+loss*alpha
        loss = k.sum(previous_loss,loss*alpha)
        return loss
    return lossFunction

from utility import *
import os
from models import predictionNn
import pandas as pd
import numpy as np

root = '/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/data/'
algo_type = "predictionNnOneEpoch"
cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
model = predictionNn(input_shape)
df = pd.read_csv('/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/test_select_data/artificial/artificialData_1.csv')
df = df.head(50)
error_prediction = []
prediction = []
L = []
convergence_loss = []
for i in np.arange(input_shape[0], len(df)):
    X_input = max_min_normalize(df["value"].values[i - (input_shape[0]):i], [])
    X_input = X_input.reshape((1,) + input_shape)
    Y_input = max_min_normalize(df["value"].values[i], [])
    Y_input = Y_input.reshape((1, 1))
    prediction.append(model.predict(X_input)[0][0])
    error_prediction.append(prediction[-1] - Y_input[0][0])
    history = model.fit(X_input, Y_input, nb_epoch=nb_epoch, verbose=0)
    convergence_loss.append(history.history['loss'])


temp_no_error = [0] * (input_shape[0])
error_prediction = temp_no_error + error_prediction
prediction = temp_no_error + prediction
L[0] = 0.5
L_no_error = [0.5] * (input_shape[0])
L = L_no_error + L
df['error_prediction'] = error_prediction
df['convergence_loss'] = temp_no_error + convergence_loss
df['prediction'] = prediction



import matplotlib.pyplot as plt
plt.plot(convergence_loss)
