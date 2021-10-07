#LIBRARIES
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model, load_model, Sequential
import tensorflow as tf
from keras.regularizers import l2



# convert an array of values into a timeseries of 3 previous steps matrix
def create_timeseries(data):
    dataX = []
    dataY = []
    for i in range(3,len(data)):
        if i%25 >= 3:
            a = np.vstack((data[i - 3], data[i - 2],data[i - 1]))
            dataX.append(a)
            dataY.append(data[i])
    return np.array(dataX), np.array(dataY)

def train_model():
    #same results for same model, makes it deterministic
    np.random.seed(1234)
    tf.random.set_seed(1234)


    #reading data
    input = np.load("../Transitions/optimal.npy", allow_pickle=True)
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
    print(data.shape)



    #reshapes trainX to be timeseries data with 3 previous timesteps
    #LSTM requires time series data, so this reshapes for LSTM purposes
    #X has 200000 samples, 3 timestep, 57 features
    inputX, inputY = create_timeseries(data)
    inputX = inputX.astype('float64')
    inputY = inputY.astype('float64')
    print(inputX.shape)
    print(inputY.shape)


    trainX = inputX[:180000]
    trainY = inputY[:180000,:90]
    valX = inputX[180000:]
    valY = inputY[180000:,:90]
    print(trainX.shape)
    print(trainY.shape)

    es1 = EarlyStopping(monitor='val_mae', mode='min', verbose=1, patience=50)
    #build functional model
    # design network
    model = Sequential()
    model.add(LSTM(128, input_shape=(trainX.shape[1],trainX.shape[2]), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(64))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # fit network
    history = model.fit(trainX, trainY, epochs=5000, batch_size=5000, verbose=2,validation_data = (valX,valY),shuffle=False,callbacks=[es1])

    model.save('Model_weights/SCTransition_model.keras')


    np.save("Model_histories/SCTransition_history.npy", history.history, allow_pickle=True)

if __name__ == '__main__':
    train_model()