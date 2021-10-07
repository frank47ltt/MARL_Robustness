#LIBRARIES
import numpy as np
from tensorflow.python.keras.models import Model, load_model, Sequential
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


num_steps = 3 - 1
threshold = .10

# convert an array of values into a timeseries of 3 previous steps matrix
def create_timeseries(dataset, y,truth):
    dataX = []
    dataY = []
    trueList = []
    for i in range(num_steps,len(dataset)):
        if i%25 >= num_steps:
            a = np.array(dataset[i-num_steps])
            for j in reversed(range(num_steps)):
                a = np.vstack((a,dataset[i - j]))
            dataX.append(a)
            dataY.append(y[i])
            trueList.append(truth[i])
    return np.asarray(dataX), np.asarray(dataY), np.asarray(trueList)


#same results for same model, makes it deterministic
np.random.seed(1234)


#reading data
input = np.load("../Datasets/datasets_nav_whiterandom/Transition_adv_25.npy", allow_pickle=True)
pre = np.asarray(input[:,0])
a0 = np.asarray(input[:,1])
a1 = np.asarray(input[:,2])
a2 = np.asarray(input[:,3])
truth = np.asarray(input[:,5])
actions = np.column_stack((a0,a1,a2))


#flattens the np arrays
pre = np.concatenate(pre).ravel()
pre = np.reshape(pre, (pre.shape[0]//54,54))


#reshapes trainX to be timeseries data with 3 previous timesteps
#LSTM requires time series data, so this reshapes for LSTM purposes
#X has 200000 samples, 3 timestep, 55 features
inputX, inputY, truth = create_timeseries(pre,actions,truth)
testX = inputX.astype('float64')
testY = inputY.astype('int64')


model0 = load_model('Agent0NetworkNav.keras')
model1 = load_model('Agent1NetworkNav.keras')
model2 = load_model('Agent2NetworkNav.keras')

pred0 = np.array(model0.predict(testX))
pred1 = np.array(model1.predict(testX))
pred2 = np.array(model2.predict(testX))

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
    result = all(elem in combined for elem in testY[i])
    if result:
        binary_anomalies.append(0)
    else:
        binary_anomalies.append(1)
binary_anomalies = np.array(binary_anomalies)

print(accuracy_score(truth,binary_anomalies))
print(confusion_matrix(truth,binary_anomalies))
print(classification_report(truth,binary_anomalies))



