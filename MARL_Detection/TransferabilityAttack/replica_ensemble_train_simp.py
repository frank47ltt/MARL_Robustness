#This file takes input, feeds into the separate agents models, then outputs probability confidences
#The probability confidences become the target outputs for our new model
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.python.keras.models import Model, load_model, Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.activations import sigmoid
import tensorflow as tf


X = np.load('query_simpX_ZS_WB.npy')
print(X.shape)

Y = np.load('query_simpY_ZS_WB.npy')
print(Y.shape)
Y = to_categorical(Y)
trainX = X[:176000]
valX = X[44000:]
trainY = Y[:176000]
valY = Y[44000:]


es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=50)

# design network
model = Sequential()
model.add(LSTM(128,input_shape=(trainX.shape[1],trainX.shape[2]),kernel_regularizer=l2(.001)))
model.add(Dense(64, activation='relu',kernel_regularizer=l2(.001)))
model.add(Dense(32, activation='relu',kernel_regularizer=l2(.001)))
model.add(Dense(16, activation='relu',kernel_regularizer=l2(.001)))
model.add(Dense(8, activation='relu',kernel_regularizer=l2(.001)))
model.add(Dense(valY.shape[1]))
model.add(Activation(sigmoid))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# fit network
history = model.fit(trainX, trainY, epochs=5000, batch_size=5000, verbose=2,validation_data = (valX,valY),shuffle=False, callbacks=[es])

model.save('ReplicaSimpEnsembleLogits_ZS_WB.keras')
print(model.summary())
