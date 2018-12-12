#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 20:47:48 2018

@author: abdulliaqat
"""

from utility import train_prediction_based_models_new,use_whole_data 
from utility import write_result,common_code_normalized, store_param
from utility import common_code
import os
from models import predictionLstm

cwd = os.getcwd()
window_size = 30
nb_epoch = 1
nb_features = 1
normalized_input = True
multistep = 1
# mse, mae or logcosh
anomalyScore_func = "mse"
anomalyScore_type = "convergence_loss"
algo_core = "prediction"
algo_type = "LSTM"
input_shape = (window_size,nb_features)
model = predictionLstm(input_shape,anomalyScore_func)
if(normalized_input):
    data_files,add_to_name, data_config = common_code_normalized()
else:
    data_files,add_to_name, data_config = common_code()
algo_name = algo_core+algo_type +"Window"+str(window_size)+anomalyScore_func+anomalyScore_type+add_to_name
result_files = use_whole_data(data_files,
                              input_shape,
                              train_prediction_based_models_new,
                              model,
                              nb_epoch=nb_epoch,
                              anomaly_score = anomalyScore_type)
print(algo_name)
write_result(algorithm_name=algo_name,data_files=result_files,
             results_path=cwd+'/results')
#store_param(window_size,nb_epoch,input_shape,algo_type,algo_name,model,
#            data_config)

store_param(window_size,nb_epoch,input_shape,algo_core,
                algo_type,algo_name,model,normalized_input,
                anomalyScore_func,anomalyScore_type,multistep
                )


#for i in range(len(df)): 
#    a.append(al.anomalyProbability(df.value.values[i],df.anomaly_score.values[i],df.timestamp.values[i]))