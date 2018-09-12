from utility import get_sample_df
import matplotlib.pyplot as plt
model_name = 'predictionNnOneEpoch7Sep'
path = '/results/{}/realKnownCause/'.format(model_name)
file = '{}_nyc_taxi.csv'.format(model_name)
df = get_sample_df(path = path,file = file)
plt.plot(df.value.values)
plt.show()
plt.plot(df.anomaly_score.values)
plt.show()
plt.plot(df.error_prediction.values)
plt.show()
