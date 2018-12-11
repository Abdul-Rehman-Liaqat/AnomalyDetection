
import argparse
from subprocess import call
import json
import pandas as pd

parser = argparse.ArgumentParser(description='Add to existing name')
parser.add_argument('--algo', help='add to existing name especially if I am testing some new feature.')
args = parser.parse_args()
algo = args.algo
with open('results/final_results.json') as data_file:
    data = json.load(data_file)
df = pd.DataFrame.from_records([dict(data[algo])])
df.to_csv('results/scores_without_postProcessing.csv', mode='a', index=False, header=False, sep=';')
call(["python","anomaly_likelihood.py","--algo",algo])
call(["python","run.py","-d",algo,"--optimize","--score","--normalize","--skipConfirmation"])
call(["python","write_anomalies.py","--algo",algo])
