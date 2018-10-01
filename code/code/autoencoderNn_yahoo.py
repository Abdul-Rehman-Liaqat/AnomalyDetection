from models import autoencoderNn
from utility import train_autoencoder_based_models,use_whole_data, write_result, common_code, store_param, get_sample_df
import os
import numpy as np
import matplotlib.pyplot as plt

window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
model = autoencoderNn(input_shape)
df = get_sample_df(path = '/code/code/data/yahoo_data/A1Benchmark/', file = 'real_10.csv' )
df = train_autoencoder_based_models(df,model,input_shape,nb_epoch=1)

df['error_prediction_squared'] = df['error_prediction'] * df['error_prediction']
def cal_threshold(val,sigma):
    mean = np.mean(val)
    std = np.std(val)
    print(mean,std)
    return mean+std*sigma


df['anomaly'] = 0
threshold = cal_threshold(df.error_prediction,5)
#df.loc[df['error_prediction_squared'] > threshold, 'anomaly'] = 1
df.loc[df['error_prediction'] > threshold, 'anomaly'] = 1


plt.plot(df.is_anomaly)
plt.plot(df.anomaly)
plt.show()



