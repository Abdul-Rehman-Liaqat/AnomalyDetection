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

from utility import use_yahoo_data, train_prediction_based_models
import os
from models import predictionNn
root = '/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/data/'
algo_type = "predictionNnOneEpoch"
cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
model = predictionNn(input_shape)
df = pd.read_csv('/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/test_select_data/artificial/artificialData_1.csv')
df = train_prediction_based_models(df,model,input_shape,nb_epoch=1)

import numpy as np
import pandas as pd
import datetime

#def artificial_data_generation(random_seed = 2):
length = 100000
sep1 = 15020
base_array = np.array([1]*length)
base_array[4000:4020] = np.random.randint(20,30,20)
base_array[7000:7020] = np.random.randint(10,20,20)
base_array[9000:9020] = np.random.randint(10,20,20)
base_array[9900:9920] = np.random.randint(20,30,20)
base_array[10000:10020] = np.random.randint(10,20,20)
base_array[11000:11020] = np.random.randint(20,30,20)
base_array[12000:12020] = np.random.randint(10,20,20)
base_array[13000:13020] = np.random.randint(10,20,20)
base_array[14000:14020] = np.random.randint(20,30,20)
base_array[15000:sep1] = np.random.randint(10,20,20)
base_array[16000:16000+sep1] = base_array[0:sep1] + 3.14
base_array[2*16000:2*16000+sep1] = base_array[0:sep1]
base_array[3*16000:3*16000+sep1] = base_array[0:sep1] + 3.14
base_array[4*16000:4*16000+sep1] = base_array[0:sep1]
df = pd.DataFrame(base_array,columns=['value'])

is_anomaly = np.array([0]*len(base_array))
is_anomaly[4000:4020] = 1
is_anomaly[7000:7020] = 1
is_anomaly[9000:9020] = 1
is_anomaly[9900:9920] = 1
is_anomaly[10000:10020] = 1
is_anomaly[11000:11020] = 1
is_anomaly[12000:12020] = 1
is_anomaly[13000:13020] = 1
is_anomaly[14000:14020] = 1
is_anomaly[15000:sep1] = 1
is_anomaly[16000:16000+sep1] = is_anomaly[0:sep1]
is_anomaly[2*16000:2*16000+sep1] = is_anomaly[0:sep1]
is_anomaly[3*16000:3*16000+sep1] = is_anomaly[0:sep1]
is_anomaly[4*16000:4*16000+sep1] = is_anomaly[0:sep1]


time = datetime.datetime.now()
def add_hours(time,h):
    return  (time + datetime.timedelta(hours=h)).strftime("%Y-%m-%d %X")

timestamp = []
for h in range(len(base_array)):
    timestamp.append(add_hours(time,h))
df['timestamp'] = timestamp
df['is_anomaly'] = is_anomaly
df.to_csv('/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/test_select_data/artificial/artificialData_1.csv',index = False)