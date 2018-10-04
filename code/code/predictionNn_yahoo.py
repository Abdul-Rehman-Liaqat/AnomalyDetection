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
use_yahoo_data(model,algo_type,input_shape,train_prediction_based_models)