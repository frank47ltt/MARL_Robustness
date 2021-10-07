#LIBRARIES
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model, load_model, Sequential
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf
from keras.utils import to_categorical


agent_to_train = 2
num_steps = 3 - 1

# convert an array of values into a timeseries of 3 previous steps matrix
def create_timeseries(dataset, y):
    dataX = []
    dataY = []
    for i in range(num_steps,len(dataset)):
        if i%25 >= num_steps:
            a = np.array(dataset[i-num_steps])
            for j in reversed(range(num_steps)):
                a = np.vstack((a,dataset[i - j]))
            dataX.append(a)
            dataY.append(y[i])
    return np.asarray(dataX), np.asarray(dataY)


#same results for same model, makes it deterministic
np.random.seed(1234)


#reading data
input = np.load("../Datasets/datasets_simp_whiterandom/Simpadv_0attack.npy", allow_pickle=True)
pre = np.asarray(input[:,0])
a0 = np.asarray(input[:,1])
a1 = np.asarray(input[:,2])
a2 = np.asarray(input[:,3])
actions = np.column_stack((a0,a1,a2))


#flattens the np arrays
pre = np.concatenate(pre).ravel()
pre = np.reshape(pre, (pre.shape[0]//28,28))


#reshapes trainX to be timeseries data with 3 previous timesteps
#LSTM requires time series data, so this reshapes for LSTM purposes
#X has 200000 samples, 3 timestep, 55 features
inputX, inputY = create_timeseries(pre,actions[:,agent_to_train])
inputX = inputX.astype('float64')
inputY = inputY.astype('int64')


trainX = inputX[:180000]
trainY = inputY[:180000]
valX = inputX[180000:]
valY = inputY[180000:]

trainY = to_categorical(trainY)
valY = to_categorical(valY)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=50)

# design network
model = Sequential()
model.add(LSTM(32, input_shape=(trainX.shape[1],trainX.shape[2]), return_sequences=True,kernel_regularizer=l2(.001)))
model.add(LSTM(16, return_sequences=True,kernel_regularizer=l2(.001)))
model.add(LSTM(10,kernel_regularizer=l2(.001)))
model.add(Dense(valY.shape[1],activation='softmax',kernel_regularizer=l2(.001)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# fit network
history = model.fit(trainX, trainY, epochs=1000, batch_size=5000, verbose=2,validation_data = (valX,valY),shuffle=False, callbacks=[es])

model.save('Agent' + str(agent_to_train)+ 'NetworkSimpRandom.keras')
print(model.summary())

np.save("agent" + str(agent_to_train) + "_history_simpRandom.npy", history.history, allow_pickle=True)