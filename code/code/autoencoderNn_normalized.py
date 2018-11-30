from utility import train_autoencoder_based_models_new,use_whole_data, write_result, store_param, common_code_normalized
import os
from models import autoencoderNn

cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
#model = autoencoderNn(input_shape,loss = 'mae')
model = autoencoderNn(input_shape)
data_files,add_to_name, data_config = common_code_normalized()
result_files = use_whole_data(data_files,
                              input_shape,
                              train_autoencoder_based_models_new,
                              model,
                              nb_epoch=nb_epoch,
                              anomaly_score = "convergence_loss",
                              config_path=data_config)
algo_type = "autoencoderNn"
algo_name = algo_type + add_to_name
print(algo_name)
write_result(algorithm_name=algo_name,data_files=result_files,results_path=cwd+'/results')
store_param(window_size,nb_epoch,input_shape,algo_type,algo_name,model,data_config)

