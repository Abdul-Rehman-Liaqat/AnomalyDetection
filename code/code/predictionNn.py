from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense
from utility import read_data,train_prediction_based_models,use_whole_data, write_result
from models import predictionNn
import os
import pickle

cwd = os.getcwd()
path = cwd + "/data"
data_files = read_data(path)
window_size = 10
nb_epoch = 80
nb_features = 1
input_shape = (window_size,)
model = predictionNn(input_shape)
result_files = use_whole_data(data_files,input_shape,train_prediction_based_models,model,nb_epoch=nb_epoch      )
with open('predictionNn_results.obj','wb') as f:
    pickle.dump(result_files,f)
write_result(algorithm_name='predictionNn',data_files=result_files,results_path=cwd+'/results')