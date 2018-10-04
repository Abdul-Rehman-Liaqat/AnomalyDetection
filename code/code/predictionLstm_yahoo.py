from models import predictionLstm
from utility import train_prediction_based_models, get_sample_df, cal_threshold, cal_auc, cal_f1score
import os
import matplotlib.pyplot as plt

cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,nb_features)
model = predictionLstm(input_shape)
df = get_sample_df(path = '/code/code/data/yahoo/', file = 'real_3.csv' )
#df = get_sample_df(path = '/code/code/data/yahoo/', file = 'real_9.csv' )
df = train_prediction_based_models(df,model,input_shape,nb_epoch=1)

df['error_prediction_squared'] = df['error_prediction'] * df['error_prediction']

df = cal_threshold(df,'error_prediction_squared',3)
auc = cal_auc(df['is_anomaly'].values,df['anomaly'].values)
f1score = cal_f1score(df['is_anomaly'].values,df['anomaly'].values)
plt.plot(df.is_anomaly)
plt.plot(df.anomaly, color = 'red')
plt.show()



