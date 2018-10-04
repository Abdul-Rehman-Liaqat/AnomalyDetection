from utility import train_autoencoder_based_models, use_yahoo_data
import os
from models import autoencoderLstm

cwd = os.getcwd()
window_size = 10
nb_epoch = 20
nb_features = 1
input_shape = (window_size, nb_features)
model = autoencoderLstm(input_shape)
algo_type = "autoencoderLstmOneEpoch"
use_yahoo_data(model,algo_type,input_shape,train_autoencoder_based_models)