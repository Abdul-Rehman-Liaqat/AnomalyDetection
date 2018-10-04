from utility import use_yahoo_data,train_autoencoder_based_models
import os
from models import autoencoderNn

cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
model = autoencoderNn(input_shape)
algo_type = "autoencoderNnOneEpoch"
use_yahoo_data(model,algo_type,input_shape,train_autoencoder_based_models)