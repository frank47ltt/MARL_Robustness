import matplotlib
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential, load_model
import tensorflow as tf


# convert an array of values into a timeseries of 4 previous steps matrix
def create_timeseries(data, truth):
    dataX = []
    dataY = []
    for i in range(3,len(data)):
        if i%25 >= 3:
            dataX.append(np.vstack((data[i - 3], data[i - 2],data[i - 1],data[i])))
            dataY.append(truth[i])
    return np.array(dataX),np.array(dataY)


#same results for same model, makes it deterministic
np.random.seed(1234)
tf.random.set_seed(1234)


#reading data
input = np.load("../Datasets/datasets_nav_blacktimed/coop_nav_blackbox_timed_25.npy", allow_pickle=True)


pre = np.asarray(input[:,0])
a1 = np.asarray(input[:,1])
a2 = np.asarray(input[:,2])
a3 = np.asarray(input[:,3])
truth = np.asarray(input[:,5])

#flattens the np arrays
pre = np.concatenate(pre).ravel()
pre = np.reshape(pre, (pre.shape[0]//54,54))

data = np.column_stack((pre,a1.T,a2.T,a3.T))

#reshapes trainX to be timeseries data with 4 previous timesteps
#LSTM requires time series data, so this reshapes for LSTM purposes
#X has 200000 samples, 4 timestep, 57 features
testX,testY= create_timeseries(data,truth)
testX = testX.astype('float64')
testY = testY.astype('int32')


model = load_model('LSTMNavBlackTimed.keras')

pred = np.array(model.predict(testX))
pred = np.argmax(pred,axis=1)

print(accuracy_score(testY,pred))
print(classification_report(testY,pred))
matrix = confusion_matrix(testY,pred)
print(matrix)
print(float(matrix[0][0])/(float(matrix[0][0])+float(matrix[0][1])))
print(float(matrix[0][1])/(float(matrix[0][0])+float(matrix[0][1])))
print(float(matrix[1][0])/(float(matrix[1][0])+float(matrix[1][1])))
print(float(matrix[1][1])/(float(matrix[1][0])+float(matrix[1][1])))

