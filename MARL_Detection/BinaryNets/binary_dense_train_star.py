import matplotlib
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
import tensorflow as tf

#same results for same model, makes it deterministic
np.random.seed(1234)
tf.random.set_seed(1234)


# convert an array of values into a timeseries of 3 previous steps matrix
def create_data(data, truth,inserted):
    dataX = []
    dataY = []
    for i in range(len(data)):
        if i%25 >= 3 and inserted[i] == 1:
            dataX.append(data[i])
            dataY.append(truth[i])
    return np.array(dataX),np.array(dataY)

#same results for same model, makes it deterministic
np.random.seed(1234)
tf.random.set_seed(1234)


#reading data
input = np.load("../Datasets/datasets_star_whitepredict/WB_SC_counterfactual_0.5.npy", allow_pickle=True)
pre = np.asarray(input[:,0])
a1 = np.asarray(input[:,1])
a2 = np.asarray(input[:,2])
a3 = np.asarray(input[:,3])
post = np.asarray(input[:,4])
truth = np.asarray(input[:,5])
inserted = np.asarray(input[:,6])

#flattens the np arrays
pre = np.concatenate(pre).ravel()
pre = np.reshape(pre, (pre.shape[0]//90,90))
post = np.concatenate(post).ravel()
post = np.reshape(post, (post.shape[0]//90,90))

data = np.column_stack((pre,a1.T,a2.T,a3.T))


#reshapes trainX to be timeseries data with 3 previous timesteps
#LSTM requires time series data, so this reshapes for LSTM purposes
#X has 200000 samples, 3 timestep, 57 features
X,Y= create_data(data,truth,inserted)
X = X.astype('float64')
print(X.shape)
print(Y.shape)


trainX = X[:120000]
trainY = Y[:120000]
valX = X[120000:]
valY = Y[120000:]
trainY = to_categorical(trainY)
valY = to_categorical(valY)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=100)

model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(valY.shape[1],activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# fit network
history = model.fit(trainX, trainY, epochs=5000, batch_size=5000, verbose=2, validation_data = (valX,valY),shuffle=False,callbacks=es)

model.save('DenseStarWhitePredict.keras')
