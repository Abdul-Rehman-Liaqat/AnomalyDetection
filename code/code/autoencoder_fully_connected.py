from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense
from utility import read_data,train_autoencoder_based_models,use_whole_data, write_result
from models import autoencoder_fully_connected
import os
import pickle

cwd = os.getcwd()
path = cwd + "/data"
data_files = read_data(path)
window_size = 10
nb_epoch = 20
nb_features = 1
model = autoencoder_fully_connected(window_size,nb_features)
result_files = use_whole_data(data_files,window_size,nb_features,train_autoencoder_based_models,model)
with open('autoencoder_fully_connected_results.obj','wb') as f:
    pickle.dump(result_files,f)
write_result(algorithm_name='autoencoder_fully_connected',data_files=result_files,results_path=cwd+'/results')