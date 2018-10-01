from models import autoencoderNn
from utility import train_autoencoder_based_models,use_whole_data, write_result, common_code, store_param
import os


cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
model = autoencoderNn(input_shape)
data_files,add_to_name, data_config = common_code()
result_files = use_whole_data(data_files,input_shape,train_autoencoder_based_models,model,nb_epoch=nb_epoch )
algo_type = "autoencoderNnOneEpoch"
algo_name = algo_type + add_to_name
print(algo_name)
write_result(algorithm_name=algo_name,data_files=result_files,results_path=cwd+'/results')
store_param(window_size,nb_epoch,input_shape,algo_type,algo_name,model,data_config)
