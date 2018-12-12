#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:26:03 2018

@author: abdulliaqat
"""

from utility import train_nStepPrediction_based_models_new
from utility import common_code,use_whole_data, write_result
from utility import common_code_normalized, store_param
import os
from models import predictionLstmStepAhead

cwd = os.getcwd()
window_size = 30
nb_epoch = 1
nb_features = 1
normalized_input = True
multistep = 1
# mse, mae or logcosh
anomalyScore_func = "mse"
anomalyScore_type = "convergence_loss"
algo_core = "predictionMultiStep"
algo_type = "LSTM"
input_shape = (window_size,nb_features)
if(normalized_input):
    data_files,add_to_name, data_config = common_code_normalized()
else:
    data_files,add_to_name, data_config = common_code()
algo_name = algo_core+algo_type +"Window"+str(window_size)+anomalyScore_func+anomalyScore_type+add_to_name
model = predictionLstmStepAhead(input_shape,multistep,loss=anomalyScore_func)


result_files = use_whole_data(data_files,
                              input_shape,
                              train_nStepPrediction_based_models_new,
                              model,
                              nb_epoch=nb_epoch,
                              anomaly_score = anomalyScore_type)
algo_name = algo_type + add_to_name
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