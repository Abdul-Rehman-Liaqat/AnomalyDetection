#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:16:05 2018

@author: abdulliaqat
"""


#for i in range(len(df)): 
#    a.append(al.anomalyProbability(df.value.values[i],df.anomaly_score.values[i],df.timestamp.values[i]))

import os
import pandas as pd
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood

def get_all_files_path(root):
    files = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(root)] for val in sublist]
    return files

algo = 'predictionNnOneEpochnormalzied30WindowSize20Nov11201656'
files = get_all_files_path('results/' + algo )
for f in files:
    if(not ('score' in f)):
        df = pd.read_csv(f)
        a = []
        al = AnomalyLikelihood()
        for i in range(len(df)): 
            a.append(al.anomalyProbability(df.value.values[i],df.anomaly_score.values[i],df.timestamp.values[i]))
        df['anomaly_score'] = a
        df.to_csv(f,index = False)