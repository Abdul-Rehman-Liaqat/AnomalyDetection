from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense
from utility import read_data,train_autoencoder_based_models_new,use_whole_data, write_result,common_code_normalized
from models import autoencoderCnn,autoencoderLstm
import os
import pickle
from datetime import datetime

now = datetime.now()

cwd = os.getcwd()
path = cwd + "/data"
data_files = read_data(path)
window_size = 50
nb_epoch = 20
nb_features = 1
input_shape = (window_size, nb_features)
model = autoencoderLstm(input_shape)
result_files,add_to_name, data_config = common_code_normalized()
result_files = use_whole_data(data_files,input_shape,train_autoencoder_based_models_new,model,nStepAhead=1,
                   anomaly_score='convergenceLoss',nb_epoch = 1
                   )

algo_name = "autoencoderCnnOneEpoch{}{}{}{}".format(now.month,now.day,now.hour,now.minute)
write_result(algorithm_name=algo_name,data_files=result_files,results_path=cwd+'/results')