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


import os
import pandas as pd
data_folder_path = '/home/abdulliaqat/Desktop/thesis/NAB original/data'

list_dir = os.listdir(data_folder_path)
list_dir.remove('README.md')
data_files_path = []
data_files_name = []
for folder in list_dir:
    if (folder != 'README.md'):
        temp_path = data_folder_path + '/' + folder + '/'
        temp_data_files_name = os.listdir(temp_path)
        temp_data_files_path = os.listdir(temp_path)
        for i, file_path in enumerate(temp_data_files_path):
            temp_data_files_path[i] = temp_path + temp_data_files_path[i]
        data_files_path.append(temp_data_files_path)
        data_files_name.append(temp_data_files_name)

cols = ['length','max_val','min_val','mean','std','data_type','file_name']
meta_df = pd.DataFrame(columns = cols)
data_files = {}
for dataset_name, path_folder, name_folder in zip(list_dir, data_files_path, data_files_name):
    temp_dir = {}
    for path, name in zip(path_folder, name_folder):
        df = pd.read_csv(path)
        length = len(df)
        max_val = max(df.value.values)
        min_val = min(df.value.values)
        mean = df.value.values.mean()
        std = df.value.values.std()
        meta_df.loc[len(meta_df)] = [length,max_val,min_val,mean,std,dataset_name,name]
meta_df.to_csv('NAB_data_meta.csv',index = False)


import os
import pandas as pd
from utility import get_all_files_path
files = get_all_files_path('/home/abdulliaqat/Desktop/thesis/yahoo')
files = [file for file in files if ('.txt' not in file)]
cols = ['length','max_val','min_val','mean','std','data_type','file_name']
meta_df = pd.DataFrame(columns = cols)
for file in files:
    df = pd.read_csv(file)
    cols = ['length','max_val','min_val','mean','std','data_type','file_name']
    length = len(df)
    max_val = max(df.value.values)
    min_val = min(df.value.values)
    mean = df.value.values.mean()
    std = df.value.values.std()
    meta_df.loc[len(meta_df)] = [length, max_val, min_val, mean, std, '/'.join(file.split('/')[0:-1]),file.split('/')[-1]]
meta_df.to_csv('Yahoo_data_meta.csv',index = False)