from models import autoencoderCnn, convert_resultjson_to_csv


window_size = 10
nb_epoch = 1
nb_features = 1
input_shape = (window_size, nb_features)


def autoencoderCnn(input_shape,loss='mse',optimizer='adam'):
#    model = Sequential()
#    model.add(Conv1D(filters=5, kernel_size=10, input_shape=input_shape, activation='relu'))
#    model.add(Reshape(target_shape=(5,1)))
#    model.add(Conv1D(filters=input_shape[0], kernel_size=5, activation='relu'))
#    model.add(Reshape(target_shape=(input_shape[0],1)))
#    model.summary()
#    model.compile(loss=loss, optimizer=optimizer)
#    return model
    model = Sequential()
    model.add(Conv1D(kernel_size=3, filters=5, strides = 2, input_shape=input_shape, activation="relu"))
    model.add(Conv1D(kernel_size=2, filters=5, input_shape=input_shape, activation="relu"))
#        model.add(Reshape(target_shape=(0, 3, 10)))
#    model.add(Flatten())
    model.add(Reshape((1,) + model.output_shape[1:]))
    print(model.output_shape)
    model.summary()
#    model.add(Dense(1))
    model.add(Conv2DTranspose(kernel_size=2, filters=5, strides = 2,  activation="relu"))
    model.add(Conv2DTranspose(kernel_size=3, filters=5, strides = 2,  activation="relu"))
    model.add(Conv2DTranspose(kernel_size=1, filters=10, activation="relu"))
    model.summary()
#    model.compile(loss=loss, optimizer=optimizer)
    return model



model = autoencoderCnn(input_shape)
df = convert_resultjson_to_csv()
df = df.iloc[[0,1,2,3,4,11,12,13,14,15,16,17,18,19,20,21]]
