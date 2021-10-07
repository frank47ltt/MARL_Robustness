#LIBRARIES
import numpy as np
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import kullback_leibler_divergence
from tensorflow.keras.losses import CategoricalCrossentropy
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt



# convert an array of values into a timeseries of 3 previous steps matrix
def create_timeseries(dataset, y):
    dataX = []
    dataY = []
    for i in range(2,len(dataset)):
        if i%25 > 1:
            a = np.concatenate((dataset[i - 2], dataset[i - 1],dataset[i]), axis=None)
            dataX.append(a)
            dataY.append(y[i])
    return np.array(dataX), np.array(dataY)


def build_model(scope, fname=None):
    with tf.variable_scope(scope):
        # build functional model
        visible = Input(shape=(3, 10))
        hidden1 = LSTM(32, return_sequences=True, name='firstLSTMLayer')(visible)
        hidden2 = LSTM(16, name='secondLSTMLayer', return_sequences=True)(hidden1)

        hidden_final = LSTM(10, name='leftBranch')(hidden2)
        agent0 = Dense(5, activation='softmax', name='agent1classifier')(hidden_final)


        model = Model(inputs=visible, outputs=agent0)

        model.compile(optimizer='adam',
                      loss={'agent1classifier': 'categorical_crossentropy'},
                      metrics={'agent1classifier': ['acc']})

        model.summary()

        if fname is not None:
            model.load_weights(fname)

    return model


def get_logits(model, obs, action, in_length):
    obs = np.reshape(obs, [1, in_length])
    action = np.reshape(action, [1, 1])
    model_input = np.concatenate([obs, action], axis=1)
    logits = np.asarray(model.predict(model_input.astype('float64'), verbose=0))
    return logits


def test_model(obs):

    model = build_model('Prediction', 'att_weights/actionMultiClassNetwork')
    result = model.predict_on_batch(obs)
    print(result)

def display_train_results(fname):
    history = np.load(fname, allow_pickle=True).item()
    plt.plot(history['acc'], label='Training Agent1 acc')
    plt.plot(history['val_acc'], label='Validation Agent1 acc')
    plt.legend()
    plt.show()

def train_model():
    #same results for same model, makes it deterministic
    np.random.seed(1234)
    #tf.random.set_seed(1234)


    #reading data
    input = np.load("Transition_simpadv_agent1_action.npy", allow_pickle=True)
    print(input.shape)
    pre = np.asarray(input[:,0])
    a0 = np.asarray(input[:,1])
    print(pre[0].shape)
    print(a0[0])
    #post = np.asarray(input[:,2])

    #flattens the np arrays
    pre = np.concatenate(pre).ravel()
    pre = np.reshape(pre, (pre.shape[0]//10,10))

    #post = np.concatenate(post).ravel()
    #post = np.reshape(post, (post.shape[0]//54,54))

    #prea1 = np.column_stack((pre,a1.T))
    #a2a3 = np.column_stack((a2.T,a3.T))

    """
    pre = pre.astype('float64')
    mean = np.mean(pre,axis=0)
    pre -= mean
    std = np.std(pre,axis=0)
    std = np.where(std==0, 1, std)
    pre /=std
    print(np.mean(pre,axis=0))
    print(np.std(pre,axis=0))
    """

    #reshapes trainX to be timeseries data with 3 previous timesteps
    #LSTM requires time series data, so this reshapes for LSTM purposes
    #X has 200000 samples, 3 timestep, 55 features
    inputX, inputY = create_timeseries(pre,a0)
    inputX = inputX.astype('float64')
    inputY = inputY.astype('int64')
    print(inputX.shape)
    inputX = np.reshape(inputX, newshape=(-1, 3, 10))
    print(inputX.shape)
    trainX = inputX[:180000]
    trainY = inputY[:180000]
    valX = inputX[180000:]
    valY = inputY[180000:]
    #testX = inputX[180000:]
    #testY = inputY[180000:]

    trainX = trainX.astype('float64')
    valX = valX.astype('float64')
    trainY = trainY.astype('float64')
    valY = valY.astype('float64')
    #testX = testX.astype('float64')
    #testY = testY.astype('float64')

    """
    print(trainX.shape)
    print(trainY.shape)
    print(valX.shape)
    print(valY.shape)
    #print(testX.shape)
    #print(testY.shape)
    print(np.unique(trainY,return_counts=True))
    """

    #two categorical arrays, one for each side of the functional network
    #testY1, testY2 = np.hsplit(testY,2)
    #testY1 = to_categorical(testY1)
    #testY2 = to_categorical(testY2)
    valY = to_categorical(valY)
    trainY = to_categorical(trainY)

    model = build_model("Agent1_Prediction")

    print(model.summary())


    history = model.fit(trainX,
                        y={'agent1classifier': trainY}, epochs=500, batch_size=5000, verbose=2,
                        validation_data = (valX,
                                           {'agent1classifier': valY}),shuffle=False)

    #save_model(model, 'actionMultiClassNetwork')
    model.save_weights('att_weights/adv_agent1_policy_predictor')


    #model = load_model("actionMultiClassNetwork.keras")

    print(history.history)
    np.save("agent1_policy_history.npy", history.history, allow_pickle=True)

if __name__ == "__main__":
    train_model()
    display_train_results("agent1_policy_history.npy")
    #obs = np.random.random(size=[1, 3, 18])
    #act = [np.random.randint(5) for i in range(3)]
    #act = np.reshape(act, [1, 3, 1])
    #inputs = np.concatenate([obs, act], axis=2)
    #print(inputs.shape)
    #test_model(inputs)