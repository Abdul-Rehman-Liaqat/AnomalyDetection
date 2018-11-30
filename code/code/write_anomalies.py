import argparse
from subprocess import call
import pandas as pd
import os

parser = argparse.ArgumentParser(description='Add to existing name')
parser.add_argument('--algo', help='add to existing name especially if I am testing some new feature.')
args = parser.parse_args()
algo = args.algo

def set_threshold(df,t,col):    
    df['predicted_anomaly'] = 0
    df.loc[df[col] > t, 'predicted_anomaly'] = 1
    return df 

def get_all_files_path(root):
    files = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(root)] for val in sublist]
    return files

def write_anomalies(algo,col = 'anomaly_score'):
    t = pd.read_csv('results/'+algo+'/'+algo+'_standard_scores.csv').Threshold.unique()[0]
    files = get_all_files_path('results/'+algo)
    for f in files:
        if(not ('_score' in f)):
            print(f)
            df = pd.read_csv(f)
            df = set_threshold(df,t,col)
            df.to_csv(f,index = False)

write_anomalies(algo)