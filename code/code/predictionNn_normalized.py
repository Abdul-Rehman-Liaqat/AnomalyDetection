from utility import train_prediction_based_models_new,use_whole_data, write_result,common_code_normalized, store_param
import os
from models import predictionNn

cwd = os.getcwd()
window_size = 40
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
model = predictionNn(input_shape,loss='mae')
#model = predictionNn(input_shape,loss='logcosh')
data_files,add_to_name, data_config = common_code_normalized()
result_files = use_whole_data(data_files,input_shape,
                              train_prediction_based_models_new,model,
                              nb_epoch=nb_epoch,
                              anomaly_score='convergence_loss')
algo_type = "predictionNnOneEpoch"
algo_name = algo_type + add_to_name
print(algo_name)
write_result(algorithm_name=algo_name,data_files=result_files,
             results_path=cwd+'/results')
store_param(window_size,nb_epoch,input_shape,algo_type,algo_name,model,
            data_config)


#for i in range(len(df)): 
#    a.append(al.anomalyProbability(df.value.values[i],df.anomaly_score.values[i],df.timestamp.values[i]))