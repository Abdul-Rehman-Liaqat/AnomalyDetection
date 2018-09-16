from utility import get_sample_df, plot_original_anomalies
import matplotlib.pyplot as plt

model_name = 'predictionNnOneEpoch9121121'
path = '/results/{}/realKnownCause/'.format(model_name)
#file = '{}_nyc_taxi.csv'.format(model_name)
file = '{}_machine_temperature_system_failure.csv'.format(model_name)
df = get_sample_df(path = path,file = file)
plt.plot(df.value.values)
plt.show()
plt.plot(df.anomaly_score.values)
plt.show()
plt.plot(df.prediction.values[200:300])
plt.plot(df.value.values[200:300])
plt.show()
plt.plot(df.anomaly_score.values)
plt.show()


dataset_name = 'realKnownCause/nyc_taxi.csv'
plot_original_anomalies(data_set=dataset_name)

