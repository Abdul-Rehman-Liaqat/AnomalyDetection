#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:26:03 2018

@author: abdulliaqat
"""

from utility import common_code,use_whole_data, write_result
from utility import common_code_normalized, store_param
import os
from models import predictionCnnStepAhead

from utility import addDummyData,get_all_files_path
from utility import movingNormal
from datetime import datetime
import numpy as np
import pandas as pd

def train_nStepPrediction_based_models_new(df,
                                           model,
                                           input_shape,
                                           nb_epoch=20,
                                           nStepAhead=1,
                                           anomaly_score="convergenceLoss"
                                           ):
    prediction = []
    convergence_loss = []
    convergence_loss_normal = []
    minVal = 0
    maxVal = 0
    W = input_shape[0]
    start_point = W+nStepAhead
    for ind,i in enumerate(np.arange(start_point,len(df))):
        X_input = df["value"].values[i - (start_point):\
                    i-nStepAhead]
        X_input = X_input.reshape((1,)+input_shape)
        Y_input = df["value"].values[i-nStepAhead:i]
        Y_input = Y_input.reshape((1,)+(nStepAhead,))
        pred = (model.predict(X_input))
        prediction.append(pred[0][0])
        history = model.fit(X_input,Y_input , nb_epoch=nb_epoch, verbose=0)
        loss = history.history['loss'][0]
        convergence_loss.append(loss)
        if(ind == 0):
            minVal = loss
            maxVal = minVal
        elif(loss < minVal):
            minVal = loss
        elif(loss > maxVal):
            maxVal = loss
        if(ind < 500):
            convergence_loss_normal.append(0.5)
        else:
            convergence_loss_normal.append(movingNormal(loss,maxVal,minVal))
    print(len(convergence_loss),len(df))
    length = input_shape[0] + nStepAhead
    df['prediction'] = addDummyData(prediction,length)
    df['convergenceLoss'] = addDummyData(convergence_loss,length)
    df['convergenceLossNormal'] = addDummyData(convergence_loss_normal,length)
    df['anomaly_score'] = df[anomaly_score]
    return df

def convertNameToWrite(name,algo_name):
    split = name.split("/")
    return algo_name+"/"+split[1]+"/"+algo_name+"_"+split[-1]



cwd = os.getcwd()
window_size = 50
nb_epoch = 1
nb_features = 1
normalized_input = True
multistep = 4
# mse, mae or logcosh
anomalyScore_func = "mse"
anomalyScore_type = "convergenceLoss"
algo_core = "predictionMultiStep"
algo_type = "CNN"
data_path = "actual_data_normalized"
input_shape = (window_size,nb_features)
all_files_path = get_all_files_path(data_path)
all_folders_path = os.listdir(data_path)
all_folders_path.remove("README.md")
now = datetime.now()
add_to_name = "{}{}{}{}".format(now.month, now.day, now.hour, now.minute)
algo_name = algo_core+algo_type+"MultiStep"+str(multistep) +"Window"+\
str(window_size)+anomalyScore_func+anomalyScore_type+add_to_name

#model = predictionLstmStepAhead(input_shape,multistep)
os.mkdir("results/"+algo_name)

for folder in all_folders_path:
    os.mkdir("results/"+algo_name+"/"+folder)

for file in all_files_path:
    if(not ".md" in file):
        df = pd.read_csv(file)
        model = predictionCnnStepAhead(input_shape,multistep)
        f = train_nStepPrediction_based_models_new(df, 
                                                   model,
                                           input_shape,
                                           nb_epoch=nb_epoch,
                                           nStepAhead=multistep,
                                           anomaly_score=anomalyScore_type)
        name_to_write = convertNameToWrite(file,algo_name)
        f.to_csv("results/"+name_to_write)
        print("results/"+name_to_write)
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

#for i in range(len(df)): 
#    a.append(al.anomalyProbability(df.value.values[i],df.anomaly_score.values[i],df.timestamp.values[i]))

# 1- Params of model
# 2- Params of training
# 3- Get model
# 4- Get type of training
# 5- Train and get result
# 6- Write output and params