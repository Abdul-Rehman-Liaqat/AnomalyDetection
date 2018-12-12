from utility import train_nStepPrediction_based_models_new,common_code,\
use_whole_data, write_result, common_code_normalized, store_param
import os
from models import predictionNnStepAhead

cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
multistep = 5
normalized_input = True
anomalyScore_func = "mse"
anomalyScore_type = "convergenceLoss"
algo_core = "predictionMultiStep"
algo_type = "NN"
if(normalized_input):
    data_files,add_to_name, data_config = common_code_normalized()
else:
    data_files,add_to_name, data_config = common_code()
algo_name = algo_core+algo_type +"Window"+str(window_size)+anomalyScore_func+\
anomalyScore_type+add_to_name


model = predictionNnStepAhead(input_shape,multistep)

result_files = use_whole_data(data_files,
                              input_shape,
                              train_nStepPrediction_based_models_new,
                              model,
                              nStepAhead=multistep,
                              nb_epoch=nb_epoch,
                              anomaly_score = anomalyScore_type)
print(algo_name)
write_result(algorithm_name=algo_name,data_files=result_files,
             results_path=cwd+'/results')
store_param(window_size,nb_epoch,input_shape,algo_core,
                algo_type,algo_name,model,normalized_input,
                anomalyScore_func,anomalyScore_type,multistep
                )


#for i in range(len(df)): 
#    a.append(al.anomalyProbability(df.value.values[i],df.anomaly_score.values[i],df.timestamp.values[i]))

# 1- Params of model
# 2- Params of training
# 3- Get model
# 4- Get type of training
# 5- Train and get result
# 6- Write output and params