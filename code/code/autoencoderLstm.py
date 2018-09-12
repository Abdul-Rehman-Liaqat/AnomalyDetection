from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense
from utility import read_data,train_autoencoder_based_models,use_whole_data, write_result
from models import autoencoderLstm
import os
import pickle
from datetime import datetime

now = datetime.now()

cwd = os.getcwd()
path = cwd + "/data"
data_files = read_data(path)
window_size = 10
nb_epoch = 20
nb_features = 1
input_shape = (window_size, nb_features)
model = autoencoderLstm(input_shape)
result_files = use_whole_data(data_files,input_shape,train_autoencoder_based_models,model)
with open('autoencoderLstm.obj','wb') as f:
    pickle.dump(result_files,f)
write_result(algorithm_name='autoencoderLstm',data_files=result_files,results_path=cwd+'/results')
algo_name = "autoencoderLstmOneEpoch{}{}{}{}".format(now.month,now.day,now.hour,now.minute)
with open(algo_name+".obj",'wb') as f:
    pickle.dump(result_files,f)
write_result(algorithm_name=algo_name,data_files=result_files,results_path=cwd+'/results')