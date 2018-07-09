#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 07:41:07 2018

@author: abdulliaqat
"""

import pandas as pd
import os

def read_data(data_folder_path):
    '''
    Function to read whole dataset using input data_folder_path and return a dictionary with key names
    as data-set folder name. The values of each key is another dictionary whose
    keys are csv file names and values are csv files read using pandas.
    '''
    list_dir = os.listdir(data_folder_path)
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




#cwd = os.getcwd()
#path = '/Desktop/thesis/NAB original/data'
#data_folder_path = cwd+path

#data_files = read_data(data_folder_path)
#write_result('TestThisShit',data_files,'/home/abdulliaqat/Desktop/thesis/NAB/results')