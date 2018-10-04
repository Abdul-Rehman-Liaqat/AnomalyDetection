from utility import  use_yahoo_data, train_prediction_based_models
import os
from models import predictionLstm

cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,nb_features)
model = predictionLstm(input_shape)
algo_type = "predictionLstm"
use_yahoo_data(model,algo_type,input_shape,train_prediction_based_models)