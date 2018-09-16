#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 07:41:07 2018

@author: abdulliaqat
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from numpy.random import seed
from configparser import ConfigParser
import json
import matplotlib.pyplot as plt
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

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

def get_sample_df(path = "/data/realKnownCause/", file = "nyc_taxi.csv"):
    cwd = os.getcwd()
    path = cwd + path
    df = pd.read_csv(path+file)
    return df

def max_min_normalize(val,max_min_var):
    if(len(max_min_var) > 0):
        val = val - max_min_var[1]
        val = val / (max_min_var[0] - max_min_var[1])
        return val
    else:
        return val

def train_prediction_based_models(df,model,input_shape,nb_epoch=20, max_min_var = []):
    error_prediction = []
    prediction = []
    L = []
    convergence_loss = []
    for i in np.arange(input_shape[0],len(df)):
        X_input = max_min_normalize(df["value"].values[i - (input_shape[0]):i], max_min_var)
        X_input = X_input.reshape((1,)+input_shape)
        Y_input = max_min_normalize(df["value"].values[i],max_min_var)
        Y_input = Y_input.reshape((1,1))
        prediction.append(model.predict(X_input)[0][0])
        error_prediction.append(prediction[-1]-Y_input[0][0])
        history = model.fit(X_input,Y_input , nb_epoch=nb_epoch, verbose=0)
        convergence_loss.append(history.history['loss'])
        L.append(score_postprocessing(error_prediction,len(error_prediction)))
    temp_no_error = [0]*(input_shape[0])
    error_prediction = temp_no_error + error_prediction
    prediction = temp_no_error + prediction
    L[0] = 0.5
    L_no_error = [0.5]*(input_shape[0])
    L = L_no_error + L
    df['error_prediction'] = error_prediction
    df['anomaly_score'] = L
    df['convergence_loss'] = temp_no_error + convergence_loss
    df['prediction'] = prediction
    return df


def train_autoencoder_based_models(df,model,input_shape,nb_epoch=20, max_min_var = []):
    error_prediction = []
    L = []
    for i in np.arange(input_shape[0],len(df)):
        X_input = max_min_normalize(df["value"].values[i-(input_shape[0]):i], max_min_var)
        X_input = X_input.reshape((1,)+input_shape)
        pred = model.predict(X_input)
        error_prediction.append(np.sqrt((pred-X_input)*(pred-X_input))[0][0])
        history = model.fit(X_input,X_input , nb_epoch=nb_epoch, verbose=0)
        L.append(score_postprocessing(error_prediction,len(error_prediction)))
    temp_no_error = [0]*(input_shape[0])
    error_prediction = temp_no_error + error_prediction
    L[0] = 0.5
    L_no_error = [0.5]*(input_shape[0])
    L = L_no_error + L
    df['error_prediction'] = error_prediction
    df['anomaly_score'] = L
    return df

def use_whole_data(data_files,input_shape,training_function,model,loss='mse',optimizer='adam',nb_epoch = 20,config_path = None):
    if(config_path != None):
        config = ConfigParser()
        config.read(config_path)

    result_files = data_files
    for key,value in data_files.items():
        for folder_key,df in value.items():
            max_min_var = []
            if(config_path != None):
                max_min_var = json.loads(config.get(key,folder_key))
            print(folder_key)
            df = training_function(df,model,input_shape,nb_epoch,max_min_var = max_min_var)
            result_files[key][folder_key] = df
    return result_files

def score_postprocessing(s,t,W=8000,w=10):
    if(t == 0):
        return 0
    def _select_window(s,W):
        s_W = s[(t-W) if(t-W >= 0) else 0 : t]
        miu_W = np.mean(s_W)
        var_W = np.var(s_W)
        return {'miu':miu_W,'var':var_W}
    W_param = _select_window(s,W)
    w_param = _select_window(s,w)
    L = 1- norm.sf((w_param['miu']-W_param['miu'])/W_param['var'])
    return L

def update_data_config(data_folder_path,config_path):
    '''
    Function to read whole dataset using input data_folder_path and return a             dictionary with key names
    as data-set folder name. The values of each key is another dictionary whose
    keys are csv file names and values are csv files read using pandas.
    '''
    config = ConfigParser()
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
        config.add_section(dataset_name)
        for path,name in zip(path_folder,name_folder):
            df = pd.read_csv(path)
            config.set(dataset_name,name,str([max(df.value.values),min(df.value.values)]))
    with open(config_path, 'w') as configfile:
        config.write(configfile)


def common_code():
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(description='Add to existing name')
    parser.add_argument('--name', help='add to existing name especially if I am testing some new feature.')
    args = parser.parse_args()
    now = datetime.now()
    add_to_name = "{}{}{}{}".format(now.month, now.day, now.hour, now.minute)
    cwd = os.getcwd()
    # path = cwd + "/code/code/data"
    path = cwd + "/data"
    data_files = read_data(path)
    if (args.name != None):
        add_to_name = args.name + add_to_name
    return data_files,add_to_name

def store_param(window_size,nb_epoch,input_shape,algo_type,algo_name,model):
    param_dict = {}
    param_dict['window_size'] = window_size
    param_dict['nb_epoch'] = nb_epoch
    param_dict['input_shape'] = input_shape
    param_dict['algo_type'] = algo_type
    param_dict['algo_name'] = algo_name
    param_dict['model'] = model.to_json()
    df = pd.DataFrame.from_dict(param_dict)
    df.to_csv('results_param.csv', mode='a', index=False, header=False, sep=';')

def plot_original_anomalies(from_index=None, from_plus=None, data_set='realKnownCause/nyc_taxi.csv',
                            path='/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/data/'):
    df =  pd.read_csv(path+data_set)
    with open('/home/abdulliaqat/Desktop/thesis/NAB/labels/combined_windows.json', 'r') as f:
        anomaly_window = json.load(f)
    if (from_index != None):
        df = df.iloc[from_index:]
    if (from_plus != None):
        df = df.iloc[:from_plus]
    windows_to_index = []
    time_stamp = list(df.timestamp.values)
    for window in anomaly_window[data_set]:
        window_index = []
        for val in window:
            if (val[0:-7] in time_stamp):
                window_index.append(time_stamp.index(val[0:-7]))
            print(val, window_index)
        if (len(window_index) == 2):
            windows_to_index.append(window_index)

    plt.plot(df.value.values)
    for window in windows_to_index:
        plt.axvspan(window[0], window[1], color='red', alpha=0.5)
    plt.show()
    return plt

#def display_algo_confusion_matrix(results_path):


#cwd = os.getcwd()
#root_path = os.path.abspath(os.path.join(cwd ,"../../.."))
#data_folder_path = root_path+'/NAB/data'
#results_folder_path = root_path+'/NAB/results'

#data_files = read_data(data_folder_path)
#write_result('TestThisShit',data_files,}}'/home/abdulliaqat/Desktop/thesis/NAB/results')