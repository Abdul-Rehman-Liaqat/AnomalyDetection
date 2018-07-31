#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 07:41:07 2018

@author: abdulliaqat
"""

import pandas as pd
import numpy as np
import os
import plotly.plotly as py



def read_data(data_folder_path):
    '''
    Function to read whole dataset using input data_folder_path and return a             dictionary with key names
    as data-set folder name. The values of each key is another dictionary whose
    keys are csv file names and values are csv files read using pandas.
    '''
    list_dir = os.listdir(data_folder_path)
    list_dir.remove('README.md')
    data_files_path = []
    data_files_name = []
    for folder in list_dir:
        if(folder != 'README.md'):        
            temp_path = data_folder_path+'/'+folder+'/'
            temp_data_files_name = os.listdir(temp_path)
            temp_data_files_path = os.listdir(temp_path)
            for i,file_path in enumerate(temp_data_files_path):
                temp_data_files_path[i] = temp_path + temp_data_files_path[i]
            data_files_path.append(temp_data_files_path)
            data_files_name.append(temp_data_files_name)            
            
    data_files = {}
    for dataset_name,path_folder,name_folder in zip(list_dir,data_files_path,data_files_name):
        temp_dir = {}
        for path,name in zip(path_folder,name_folder):
            temp_dir[name] = pd.read_csv(path)
            
        data_files[dataset_name] = temp_dir
        
    return(data_files)

#def iterate_data(data_files):
#    '''
#    Function to iterate through each dataset and receive result from the algorithm
#    '''
    
    
#def preprocess_data(df):
    
def write_result(algorithm_name,data_files,results_path):
    '''
    Function to receive results from algorithms and write them into result directory
    '''
    algo_folder = results_path+'/'+algorithm_name
    os.mkdir(algo_folder)
    for key in list(data_files.keys()):
        os.mkdir(algo_folder+'/'+key)
        for file_key in list(data_files[key].keys()):
            df = data_files[key][file_key] 
            df.to_csv(algo_folder+'/'+key+'/'+algorithm_name+'_'+file_key,index = False)


#def display_algo_prediction_label(results_path):
def _get_result_folder_structure(path,parent_folder_name):
    list_all = os.listdir(path)
    attach_path = lambda x,path:[path+'/'+folder for folder in list_all]
    get_dir = lambda x:[name for name in x if(os.path.isdir(path+'/'+name))]
    get_file = lambda x:[name for name in x if(os.path.isfile(path+'/'+name))]
    list_dir = get_dir(list_all)
    list_file = get_file(list_all)
    structure = {}
    structure[parent_folder_name+'files'] = list_file
    for folder in list_dir:
        sub_path = path+'/'+folder+'/'
        sub_list_all = os.listdir(sub_path)
        sub_list_dir = get_dir(sub_list_all)
        sub_list_file = get_file(sub_list_all)
        sub_files_path = [sub_path+sub_folder for sub_folder in sub_list_dir]
        for i,file_path in enumerate(sub_list_dir):
            temp_files_path[i] = temp_path + temp_files_path[i]
        files_path.append(temp_files_path)
        files_name.append(temp_files_name)            
            
    files = {}
    for name,path_folder,name_folder in zip(list_dir,files_path,files_name):
        temp_dir = {}
        for path_file,name in zip(path_folder,name_folder):
            temp_dir[name] = pd.read_csv(path_file)
            
        files[name] = temp_dir
    files['main_folder_files'] = list_file
    return(files)

def get_sample_df():
    cwd = os.getcwd()
    path = cwd + "/data/realKnownCause/"
    df = pd.read_csv(path+"nyc_taxi.csv")
    return df


def train_prediction_based_models(df,model,input_shape,nb_epoch):
    error_prediction = []
    for i in np.arange(11,len(df)):
        X_input = df["value"].values[i-(1+input_shape[0]):i-1].reshape((1,)+input_shape)
        Y_input = df["value"].values[i].reshape((1,1))
        history = model.fit(X_input,Y_input , nb_epoch=20, verbose=0)
        error_prediction.append((model.predict(X_input)-Y_input)[0][0])
        print(i)
    temp_no_error = [0]*11
    error_prediction = temp_no_error + error_prediction
    df['anomaly_score'] = error_prediction
    return df


def train_autoencoder_based_models(df,model,input_shape,nb_epoch):
    error_prediction = []
    for i in np.arange(11,len(df)):
        X_input = df["value"].values[i-(1+input_shape[0]):i-1].reshape((1,)+input_shape)
        history = model.fit(X_input,X_input , nb_epoch=20, verbose=0)
        error_prediction.append((model.predict(X_input)-X_input)[0][0])
        print(i)
    temp_no_error = [0]*11
    error_prediction = temp_no_error + error_prediction
    df['anomaly_score'] = error_prediction
    return df



def use_whole_data(data_files,input_shape,training_function,model,loss='mse',optimizer='adam',nb_epoch = 20):
    result_files = data_files
    for key,value in data_files.items():
        for folder_key,df in value.items():
            df = training_function(df,model,input_shape,nb_epoch)
            result_files[key][folder_key] = df
    return result_files


#def display_algo_confusion_matrix(results_path):


#cwd = os.getcwd()
#root_path = os.path.abspath(os.path.join(cwd ,"../../.."))
#data_folder_path = root_path+'/NAB/data'
#results_folder_path = root_path+'/NAB/results'

#data_files = read_data(data_folder_path)
#write_result('TestThisShit',data_files,}}'/home/abdulliaqat/Desktop/thesis/NAB/results')