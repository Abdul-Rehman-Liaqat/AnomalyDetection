from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dropout, Dense
from utility import get_sample_df, train_prediction_based_models
from models import predictionNn
import os
import pickle
import numpy as np
df = get_sample_df()
window_size = 10
nb_epoch = 2
nb_features = 1
input_shape = (window_size,)
model = predictionNn(input_shape)
#df = train_prediction_based_models(df,model,input_shape,probation_period=750)
train_prediction_based_models(df,model,input_shape,750,nb_epoch=2)
# error_prediction = []
# prediction = []
# L = []
# for i in np.arange(input_shape[0]+1,len(df)):
#     X_input = df["value"].values[i-(1+input_shape[0]):i-1].reshape((1,)+input_shape)
#     Y_input = df["value"].values[i].reshape((1,1))
#     prediction.append(model.predict(X_input)[0][0])
#     error_prediction.append(prediction[-1]-Y_input[0][0])
#     history = model.fit(X_input,Y_input , nb_epoch=nb_epoch, verbose=0)
#     L.append(history.history['loss'])
#    L.append(score_postprocessing(error_prediction,len(error_prediction)))
#        print(i)
#    temp_no_error = [0]*(input_shape[0]+1)
#    error_prediction = temp_no_error + error_prediction
#    error_prediction[0:probation_period] = [0]*probation_period
#    df['anomaly_score'] = error_prediction
#    df['anomaly_prediction'] = prediction
#    return df