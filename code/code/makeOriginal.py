import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Add to existing name')
parser.add_argument('--algo', help='add to existing name especially if I am testing some new feature.')
args = parser.parse_args()
algo = args.algo

def get_all_files_path(root):
    files = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(root)] for val in sublist]
    return files


files = get_all_files_path('results/' + algo )
for f in files:
    if(not ('_score' in f)):
        print(f)
        df = pd.read_csv(f)
        df['anomaly_score'] = df['convergenceLoss']
        df.to_csv(f,index = False)