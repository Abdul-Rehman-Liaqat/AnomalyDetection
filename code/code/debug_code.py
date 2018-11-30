from utility import *
import os
'''
f = open('/home/abdulliaqat/Desktop/thesis/archive/NAB/labels/combined_windows.json', 'r')
file = json.load(f)
path = '/home/abdulliaqat/Desktop/thesis/archive/NAB/data'
path_new = '/home/abdulliaqat/Desktop/thesis/archive/NAB/data_new'
for fi in list(file.keys()):
    df = pd.read_csv(path+'/'+fi)
    df['is_anomaly'] = 0
    for time_range in file[fi]:
        df.loc[(df['timestamp'] >= time_range[0].split('.')[0]) & (df['timestamp'] <= time_range[1].split('.')[0]),'is_anomaly'] = 1
    df.to_csv(path_new+'/'+fi,index = False)
'''
'''
Numenta Baselines
algo_list = ['numenta',
            'random',
            'skyline',
            'bayesChangePt',
            'windowedGaussian',
            'expose',
            'relativeEntropy']


path = '/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/results'
labeled_data = '/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/actual_data'
n_sigma = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]

col = 'anomaly_score'
data_folder = 'yahoo'
algo_auc = []
for algo in algo_list:
    files_path = path+'/'+algo+'/'+ data_folder
    files = os.listdir(files_path)
    fi_auc = []
    for fi in files:
        fi_path = files_path + '/' + fi
        fi_name = '_'.join(fi.split('_')[1:])
        labeled_df = pd.read_csv(labeled_data+'/'+data_folder+'/'+fi_name)
        df = pd.read_csv(fi_path)
        sig_auc = []
        df['is_anomaly'] = labeled_df['is_anomaly']
        for sig in n_sigma:
            df = cal_threshold(df,col,sig)
            sig_auc.append(cal_auc(df['is_anomaly'].values,df['prediction'].values))
        fi_auc.append(sig_auc)
    fi_auc_df = pd.DataFrame(np.array(fi_auc),columns = n_sigma)
    algo_auc.append(fi_auc_df.mean())
algo_auc_df = pd.DataFrame(np.array(algo_auc),columns = n_sigma)
'''

from utility import train_autoencoder_based_models_new,use_whole_data, write_result, store_param, common_code_normalized
import os
from models import autoencoderNn

cwd = os.getcwd()
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
#model = autoencoderNn(input_shape,loss = 'mae')
model = autoencoderNn(input_shape)
df = load_result_file('autoencoderNnnormalized10WindowMAEAveraged11292252',file = 'realAWSCloudwatch/grok_asg_anomaly.csv')
error_prediction = []
convergence_loss = []
sigmoid_loss = []
for i in np.arange(len(df) - input_shape[0]):
    X_input = df["value"].values[i:i+(input_shape[0])]
    X_input = X_input.reshape((1,)+input_shape)
    pred = model.predict(X_input)
    history = model.fit(X_input,X_input , nb_epoch=nb_epoch, verbose=0)
    error_prediction.append(np.sum((np.abs(pred-X_input)[0][0]))/input_shape[0])
    convergence_loss.append(history.history['loss'][0])
    sigmoid_loss.append(sigmoid(error_prediction[-1]))
temp_no_error = [error_prediction[0]]*(input_shape[0])
error_prediction = temp_no_error + error_prediction
df['error_prediction'] = error_prediction
df['convergence_loss'] = [convergence_loss[0]]*(input_shape[0]) + convergence_loss
df['sigmoid_error_prediction'] = [sigmoid_loss[0]]*(input_shape[0]) + sigmoid_loss
df['anomaly_score'] = df['convergence_loss']

plt.plot(error_prediction)
plt.plot(df.value.values)