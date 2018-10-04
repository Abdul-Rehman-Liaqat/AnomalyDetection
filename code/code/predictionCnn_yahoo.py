from utility import train_prediction_based_models, get_all_files_path, use_yahoo_data
import os
from models import predictionCnn
root = '/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/data/'
algo_type = "predictionCnnOneEpoch"
cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size, nb_features)
model = predictionCnn(input_shape)

use_yahoo_data(model,algo_type,input_shape,train_prediction_based_models)