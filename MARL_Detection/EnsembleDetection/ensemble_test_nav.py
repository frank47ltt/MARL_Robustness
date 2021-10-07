import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential, load_model
import tensorflow as tf

num_steps = 3 - 1
threshold = .10



# convert an array of values into a timeseries of 4 previous steps matrix
def create_timeseries(data, truth):
    dataX = []
    dataY = []
    for i in range(3,len(data)):
        if i%25 >= 3:
            dataX.append(np.vstack((data[i - 3], data[i - 2],data[i - 1],data[i])))
            dataY.append(truth[i])
    return np.array(dataX),np.array(dataY)

#create timeseries for leftover data x needs (samples,num_steps,features54) and y needs actions
def leftover_timeseries(indices,testX,truth):
    dataX = []
    dataY = []
    trueList = []
    for i in range(len(indices)):
        dataX.append(testX[indices[i],:,:54])
        dataY.append(testX[indices[i],:,54:])
        trueList.append(truth[indices[i]])
    return np.asarray(dataX), np.asarray(dataY), np.asarray(trueList)


#same results for same model, makes it deterministic
np.random.seed(1234)
tf.random.set_seed(1234)


#reading data
input = np.load("../Datasets_Frank/CN_ZS_WB_100_train.npy", allow_pickle=True)

# train using 50%
# test using different a new 50% testing data set


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


binary_model = load_model('../BinaryNets/LSTMNavWhitePredict_ZS_WB.keras')

pred = np.array(binary_model.predict(testX))
pred = np.argmax(pred,axis=1)

print(accuracy_score(testY,pred))
print(classification_report(testY,pred))
print(testY.shape)
matrixA = confusion_matrix(testY,pred)
print(matrixA)

indices_fn = []
for i in range(len(pred)):
    if pred[i] == 0: #and testY[i] == 1:
        #all data predicted negative
        indices_fn.append(i)
indices_fn = np.asarray(indices_fn)
print(indices_fn.shape)


#now time to clean up with predictive models
#create timeseries for leftover data
newX, newY, newTruth = leftover_timeseries(indices_fn,testX,testY)

model0 = load_model('../SeparateAgents/Agent0NetworkNav.keras')
model1 = load_model('../SeparateAgents/Agent1NetworkNav.keras')
model2 = load_model('../SeparateAgents/Agent2NetworkNav.keras')

pred0 = np.array(model0.predict(newX))
pred1 = np.array(model1.predict(newX))
pred2 = np.array(model2.predict(newX))

pred0 = np.argwhere(pred0 >= threshold)
pred0 = np.split(pred0[:,1], np.unique(pred0[:, 0], return_index=True)[1][1:])

pred1 = np.argwhere(pred1 >= threshold)
pred1 = np.split(pred1[:,1], np.unique(pred1[:, 0], return_index=True)[1][1:])

pred2 = np.argwhere(pred2 >= threshold)
pred2 = np.split(pred2[:,1], np.unique(pred2[:, 0], return_index=True)[1][1:])

#anomaly detection
binary_anomalies = []
for i in range(len(pred0)):
    combined = np.unique(np.concatenate((pred0[i],pred1[i],pred2[i]),axis=0).ravel())
    result = all(elem in combined for elem in newY[i,-1])
    if result:
        binary_anomalies.append(0)
    else:
        binary_anomalies.append(1)
binary_anomalies = np.array(binary_anomalies)

print(accuracy_score(newTruth,binary_anomalies))
print(classification_report(newTruth,binary_anomalies))
matrix = confusion_matrix(newTruth,binary_anomalies)
print(matrix)

precision = (matrix[1][1] + matrixA[1][1]) / (matrix[1][1] + matrixA[1][1] + matrixA[0][1] +matrix[0][1])
acc = (matrixA[0][0] + matrix[1][1]  + matrixA[1][1])/testX.shape[0]
fp = matrix[0][1]/(matrixA[0][0]+matrixA[0][1])
print(" A:  " + str(acc))
print(" P:  " + str(precision))
print("FP:  " + str(fp))
print("FN:  " + str(float(matrix[1][0])/(float(matrixA[1][0])+float(matrixA[1][1]))))
print("TP:  " + str((float(matrix[1][1]) + float(matrixA[1][1]))/(float(matrixA[1][0])+float(matrixA[1][1]))))
