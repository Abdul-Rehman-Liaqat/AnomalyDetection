import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import mlflow

def get_sample_df():
    cwd = os.getcwd()
    path = cwd + "/data/realKnownCause/"
    df = pd.read_csv(path+"nyc_taxi.csv")
    return df


