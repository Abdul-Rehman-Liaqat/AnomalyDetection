import pandas as pd
import numpy as np
import sys
import os
import time
from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense
import mlflow
sys.path.insert(0, '/executor/execution_code')
from utility import read_data,write_result

cwd = os.getcwd()
path = os.path.abspath(os.path.join(cwd,"../data"))
data_files = read_data(path)

result_files = data_files
for key,value in data_files.items():
    for folder_key,df in value.items():
        print(folder_key,key)
        nb_features = 1
        window_size = 10
        model = Sequential()
        model.add(Conv1D(nb_filter=5, kernel_size=10, input_shape=(window_size, nb_features), activation='relu'))
        model.add(Flatten())
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.summary()
        model.compile(loss='mse', optimizer='adam')
        error_prediction = []
        for i in np.arange(11,len(df)):
            print(i)
            L = 10 #window size
            X_input = df["value"].values[i-(1+L):i-1].reshape((1,10,1))
            Y_input = df["value"].values[i].reshape((1,1))
            history = model.fit(X_input,Y_input , epochs=20, verbose=0)
            error_prediction.append((model.predict(X_input)-Y_input)[0][0])
        temp_no_error = [0]*11
        error_prediction = temp_no_error + error_prediction
        df['anomaly_score'] = error_prediction
        result_files[key][folder_key] = df

algorithm_name = "cnn_based_anomaly_detection"

write_result(algorithm_name=algorithm_name,data_files=result_files,results_path='/executor/execution_code/results')



        
        
