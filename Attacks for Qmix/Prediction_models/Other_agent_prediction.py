#LIBRARIES
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model,load_model
import tensorflow as tf
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.losses import kullback_leibler_divergence
from keras.losses import CategoricalCrossentropy
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix,accuracy_score



# convert an array of values into a timeseries of 3 previous steps matrix
def create_timeseries(x,y,inserted):
    dataX = []
    dataY = []
    for i in range(2,len(x)):
        if i%25 >= 2 and inserted[i] == 1 :
            dataX.append(np.vstack((x[i - 2], x[i - 1],x[i])))
            dataY.append(y[i])
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

    prea1 = np.column_stack((pre,a1.T))
    a2a3 = np.column_stack((a2.T,a3.T))
    print(prea1.shape)
    print(a2a3.shape)

    #reshapes trainX to be timeseries data with 3 previous timesteps
    #LSTM requires time series data, so this reshapes for LSTM purposes
    #X has 200000 samples, 3 timestep, 57 features
    inputX, inputY = create_timeseries(prea1,a2a3,inserted)
    inputX = inputX.astype('float64')
    inputY = inputY.astype('float64')
    print(inputX.shape)
    print(inputY.shape)


    trainX = inputX[:180000]
    trainY = inputY[:180000]
    valX = inputX[180000:]
    valY = inputY[180000:]

    valY1, valY2 = np.hsplit(valY,2)
    valY1 = to_categorical(valY1)
    valY2 = to_categorical(valY2)
    trainY1, trainY2 = np.hsplit(trainY,2)
    trainY1 = to_categorical(trainY1)
    trainY2 = to_categorical(trainY2)


    es1 = EarlyStopping(monitor='val_agent1classifier_acc', mode='max', verbose=1, patience=100)
    es2 = EarlyStopping(monitor='val_agent2classifier_acc', mode='max', verbose=1, patience=100)
    #build functional model
    visible =Input(shape=(trainX.shape[1],trainX.shape[2]))
    hidden1 = LSTM(128, return_sequences=True)(visible)
    hidden2 = LSTM(64,return_sequences=True)(hidden1)
    #first agent branch
    hiddenAgent1 = LSTM(16, name='firstBranch')(hidden2)
    agent1 = Dense(valY1.shape[1],activation='softmax',name='agent1classifier')(hiddenAgent1)
    #second agent branch
    hiddenAgent2 = LSTM(16, name='secondBranch')(hidden2)
    agent2 = Dense(valY2.shape[1],activation='softmax',name='agent2classifier')(hiddenAgent2)


    model = Model(inputs=visible,outputs=[agent1,agent2])

    model.compile(optimizer='adam',
                  loss={'agent1classifier': 'categorical_crossentropy',
                      'agent2classifier': 'categorical_crossentropy'},
                  metrics={'agent1classifier': ['acc'],
                           'agent2classifier': ['acc']})
    print(model.summary())


    history = model.fit(trainX,
                        y={'agent1classifier': trainY1,
                           'agent2classifier':trainY2}, epochs=5000, batch_size=5000, verbose=2,
                        validation_data = (valX,
                                           {'agent1classifier': valY1,
                                            'agent2classifier': valY2}),shuffle=False,callbacks=[es1,es2])

    model.save('Model_weights/SC_other_agent_predictions.keras')

if __name__ == '__main__':
    train_model()