#this one goes binary -> predictive, accuracy on 0:0.8640827919915162
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential, load_model
import tensorflow as tf

num_steps = 3 - 1
threshold = .10


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
input = np.load("../Datasets/datasets_star_whitepredict/WB_SC_counterfactual_0.25.npy", allow_pickle=True)
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
testX,testY= create_timeseries(data,truth,inserted)
testX = testX.astype('float64')
print(testX.shape)
print(testY.shape)

model = load_model('LSTMStarWhitePredict.keras')


pred = np.array(model.predict(testX))
pred = np.argmax(pred,axis=1)

print(accuracy_score(testY,pred))
print(classification_report(testY,pred))
matrix = confusion_matrix(testY,pred)
print(matrix)
if not(matrix[0][0]+matrix[0][1] == 0):
    print(float(matrix[0][0])/(float(matrix[0][0])+float(matrix[0][1])))
    print(float(matrix[0][1])/(float(matrix[0][0])+float(matrix[0][1])))
print(float(matrix[1][0])/(float(matrix[1][0])+float(matrix[1][1])))
print(float(matrix[1][1])/(float(matrix[1][0])+float(matrix[1][1])))

