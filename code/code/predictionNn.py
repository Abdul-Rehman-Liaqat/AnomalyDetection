from utility import read_data,train_prediction_based_models,use_whole_data, write_result
from models import predictionNn
import os
import pickle
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description= 'Add to existing name' )
parser.add_argument('--name', help = 'add to existing name especially if I am testing some new feature.')
args = parser.parse_args()
now = datetime.now()
cwd = os.getcwd()
#path = cwd + "/code/code/data"
path = cwd + "/data"
data_files = read_data(path)
window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size,)
model = predictionNn(input_shape)
result_files = use_whole_data(data_files,input_shape,train_prediction_based_models,model,nb_epoch=nb_epoch)
algo_name = "predictionNnOneEpoch"
if(args.name != None):
    algo_name = algo_name + args.name
algo_name = algo_name + "{}{}{}{}".format(now.month,now.day,now.hour,now.minute)
print(algo_name)
with open("dump/"+algo_name+".obj",'wb') as f:
    pickle.dump(result_files,f)
write_result(algorithm_name=algo_name,data_files=result_files,results_path=cwd+'/results')