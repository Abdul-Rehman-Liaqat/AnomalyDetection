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


algo_list = ['autoencoderLstmOneEpoch',
            'autoencoderNnOneEpoch',
            'predictionCnnOneEpoch',
            'predictionLstm',
            'predictionNnOneEpoch']


path = '/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/yahoo_data_result'
labeled_data = '/home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/code/actual_data'
n_sigma = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]

col = 'error_prediction'
data_folder = 'yahoo'
algo_auc = []
for algo in algo_list:
    files_path = path+'/'+algo
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

def cal_threshold(df,col,sigma = 5):
    df['prediction'] = 0
    val = df[col]
    val = [split_brackets(v) for v in val]
    df[col+'float_conversion'] = val
    mean = np.mean(val)
    std = np.std(val)
    threshold = mean + std * sigma
    df.loc[df[col+'float_conversion'] > threshold, 'prediction'] = 1
    return df

def split_brackets(v):
    if(type(v) == str):
        if('[' in v):
            return float(v.split('[')[1].split(']')[0])
        else:
            return float(v)
    else:
        return v


