
from keras.utils import to_categorical
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM,Dense
from tensorflow.python.keras.models import Sequential
import tensorflow as tf


# convert an array of values into a timeseries of 3 previous steps matrix
def create_timeseries(data, truth,inserted):
    dataX = []
    dataY = []
    for i in range(3,len(data)):
        if i%25 >= 3 and inserted[i] == 1:
            dataX.append(np.vstack((data[i - 3], data[i - 2],data[i - 1],data[i])))
            dataY.append(truth[i])
    return np.array(dataX),np.array(dataY)


#same results for same model, makes it deterministic
np.random.seed(1234)
tf.random.set_seed(1234)


#reading data
input = np.load("../Datasets_Frank/SC_CF_BB_train.npy", allow_pickle=True)
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
inputX,inputY= create_timeseries(data,truth,inserted)
inputX = inputX.astype('float64')
print(inputX.shape)
print(inputY.shape)
print(np.unique(truth,return_counts=True))
"""
trainX = inputX[:120000]
trainY = inputY[:120000]
valX = inputX[120000:]
valY = inputY[120000:]
"""
trainX = inputX[:170000]
trainY = inputY[:170000]
valX = inputX[40000:]
valY = inputY[40000:]
trainY = to_categorical(trainY)
valY = to_categorical(valY)




es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=100)
model = Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(trainX.shape[1],trainX.shape[2])))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
#number nodes in this layer corresponds to agent's possible decisions:it can go to 0,1,or 2 (MULTI-CLASS CLASSIFICATION)
model.add(Dense(valY.shape[1],activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# fit network
history = model.fit(trainX, trainY, epochs=5000, batch_size=5000, verbose=2, validation_data = (valX,valY),shuffle=False,callbacks=es)

model.save('LSTMStarWhitePredict_CF_BB.keras')