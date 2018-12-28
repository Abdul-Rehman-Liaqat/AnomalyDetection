import os
import pandas as pd

def get_all_files_path(root):
    files = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(root)] for val in sublist]
    return files

algo = "predictionMultiStepLSTMMultiStep3Window20mseconvergenceLoss12212156"
files = get_all_files_path('results/' + algo )
files = [e for e in files if("score" not in e)]
meanVal = []
varVal = []
for f in files:
    if(not ('_score' in f)):
        df = pd.read_csv(f)
        meanVal.append(df.convergenceLoss.mean())
        varVal.append(df.convergenceLoss.var())
meanVal = [e for e in meanVal if(not np.isnan(e))]
varVal = [e for e in varVal if(not np.isnan(e))]