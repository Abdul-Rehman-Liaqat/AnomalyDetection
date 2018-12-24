import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Add to existing name')
parser.add_argument('--algo', help='add to existing name especially if I am testing some new feature.')
parser.add_argument('--start', help='add to existing name especially if I am testing some new feature.')
args = parser.parse_args()
algo = args.algo
#start = int(args.start)
start = int(args.start)

def get_all_files_path(root):
    files = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(root)] for val in sublist]
    return files

def movingNormal(val,max,min):
    return (val-min+0.00001)/(max-min+0.00001)

            
def differentInitialPointNormalization(df,start):
    minVal = 0
    maxVal = 0
    l = []
    for ind,loss in enumerate(df.convergenceLoss):
        if(ind == 0):
            minVal = loss
            maxVal = minVal
        elif(loss < minVal):
            minVal = loss
        elif(loss > maxVal):
            maxVal = loss
            
        if(ind < start):
            l.append(0.5)
        else:
            l.append(movingNormal(loss,maxVal,minVal))        
    df["convergenceLossNormal"+str(start)] = l
    return df

files = get_all_files_path('results/' + algo )
for f in files:
    if(not ('_score' in f)):
        print(f)
        df = pd.read_csv(f)
        a = []
        df = differentInitialPointNormalization(df,start)
        df["anomaly_score"] = df["convergenceLossNormal"+str(start)]
        df.to_csv(f,index = False)        
