from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import numpy as np
import pickle

'''
https://scikit-learn.org/stable/modules/outlier_detection.html
'''

#same results for same model, makes it deterministic
np.random.seed(1234)

#reading data
input = np.load("../datasets_nav_strategic/Coopnav_100timed_attack.npy", allow_pickle=True)

pre = np.asarray(input[:,0])
a1 = np.asarray(input[:,1])
a2 = np.asarray(input[:,2])
a3 = np.asarray(input[:,3])
testY = np.asarray(input[:,5])

#flattens the np arrays
pre = np.concatenate(pre).ravel()
pre = np.reshape(pre, (pre.shape[0]//54,54))
testX = np.column_stack((pre,a1.T,a2.T,a3.T))
testX = testX.astype('float64')
testY = testY.astype('int32')

model = pickle.load(open('../KNearestAnomalyNavStrategic.sav', 'rb'))
pred = model.predict(testX)
pred = np.array(pred)

print(accuracy_score(testY,pred))
print(classification_report(testY,pred))
matrix = confusion_matrix(testY,pred)
print(matrix)
print(float(matrix[0][0])/(float(matrix[0][0])+float(matrix[0][1])))
print(float(matrix[0][1])/(float(matrix[0][0])+float(matrix[0][1])))
print(float(matrix[1][0])/(float(matrix[1][0])+float(matrix[1][1])))
print(float(matrix[1][1])/(float(matrix[1][0])+float(matrix[1][1])))


import winsound
frequency = 2000  # Set Frequency To 2500 Hertz
duration = 2000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)