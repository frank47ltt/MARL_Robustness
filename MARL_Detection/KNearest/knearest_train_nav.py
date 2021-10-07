import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import pickle

#same results for same model, makes it deterministic
np.random.seed(1234)

#reading data
input = np.load("../datasets_nav_strategic/Coopnav_50timed_attack.npy", allow_pickle=True)


pre = np.asarray(input[:,0])
a1 = np.asarray(input[:,1])
a2 = np.asarray(input[:,2])
a3 = np.asarray(input[:,3])
Y = np.asarray(input[:,5],dtype='int64')

#flattens the np arrays
pre = np.concatenate(pre).ravel()
pre = np.reshape(pre, (pre.shape[0]//54,54))

X = np.column_stack((pre,a1.T,a2.T,a3.T))
X = X.astype('float64')

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X,Y)

pickle.dump(classifier, open('../KNearestAnomalyNavStrategic.sav', 'wb'))
