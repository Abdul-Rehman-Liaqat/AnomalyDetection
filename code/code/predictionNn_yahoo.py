from utility import train_prediction_based_models,use_whole_data, write_result, common_code, store_param, get_sample_df
import os
from models import predictionNn
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
model = predictionNn(input_shape)
df = get_sample_df(path = '/code/code/data/yahoo_data/A1Benchmark/', file = 'real_1.csv' )
df = train_prediction_based_models(df,model,input_shape,nb_epoch=1)

df['error_prediction_squared'] = df['error_prediction'] * df['error_prediction']
def cal_threshold(df,col,sigma = 5):
    df['anomaly'] = 0
    val = df[col]
    mean = np.mean(val)
    std = np.std(val)
    threshold = mean + std * sigma
    df.loc[df[col] > threshold, 'anomaly'] = 1
    return df

df = cal_threshold(df,'error_prediction_squared',8)

plt.plot(df.is_anomaly)
plt.plot(df.anomaly)
plt.show()

def cal_auc(y,pre):
    from sklearn.metrics import roc_curve,auc
    fpr, tpr, thresholds = roc_curve(y,pre)
    return auc(fpr, tpr)
def cal_f1score(y,pre):
    from sklearn.metrics import f1_score
    return f1_score(y,pre)

