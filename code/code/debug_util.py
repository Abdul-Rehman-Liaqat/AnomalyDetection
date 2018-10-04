from utility import get_sample_df, plot_original_anomalies
import matplotlib.pyplot as plt

model_name = 'predictionNnOneEpoch9121121'
path = '/results/{}/realKnownCause/'.format(model_name)
#file = '{}_nyc_taxi.csv'.format(model_name)
file = '{}_machine_temperature_system_failure.csv'.format(model_name)
df = get_sample_df(path = path,file = file)
plt.plot(df.value.values)
plt.show()
plt.plot(df.anomaly_score.values)
plt.show()
plt.plot(df.prediction.values[200:300])
plt.plot(df.value.values[200:300])
plt.show()
plt.plot(df.anomaly_score.values)
plt.show()


dataset_name = 'realKnownCause/nyc_taxi.csv'
plot_original_anomalies(data_set=dataset_name)



from dateutil import parser
from datetime import timedelta, datetime
import os
import pandas as pd
#datetime.datetime.fromtimestamp(1485714600)




path = '/home/abdulliaqat/Desktop/thesis/yahoo/A1Benchmark'
dir = os.listdir(path)
dir = [i for i in dir if('txt' not in i)]
for d in dir:
    dir_path = path+'/'+d
    df = pd.read_csv(dir_path)
    val = df['timestamp'].values
    original_time = '2014-04-01 00:00:00'
    original = parser.parse(original_time)
    converted = []
    for i in val:
        converted.append(str(original + timedelta(hours = int(i))))
    df['timestamp'] = converted
    df.to_csv(dir_path, index = False)


algos = ['null','numenta','random','skyline','bayesChangePt','windowedGaussian','expose','relativeEntropy']
file = ['real_','synthetic_']

original_path = {'real':'/home/abdulliaqat/Desktop/thesis/yahoo/A1Benchmark/',
                 'synthetic':'/home/abdulliaqat/Desktop/thesis/yahoo/A2Benchmark/'}

import pandas as pd
from utility import cal_threshold,cal_auc,cal_f1score
import matplotlib.pyplot as plt

algo_name = algos[1]
i = 1
data_type = 'real'
path  = '/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/results/'
df = pd.read_csv(path+algo_name+'/'+'yahoo/'+algo_name+'_'+data_type+'_'+str(i)+'.csv')
df_original = pd.read_csv(original_path[data_type]+data_type+'_'+str(i)+'.csv')
df['is_anomaly'] = df_original['is_anomaly']
df = cal_threshold(df,'anomaly_score')
print(cal_f1score(df['is_anomaly'].values,df['anomaly'].values))
print(cal_auc(df['is_anomaly'].values,df['anomaly'].values))
plt.plot(df.is_anomaly)
#plt.plot(df.anomaly)
plt.show()

