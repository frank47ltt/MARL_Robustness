# evaluate random forest algorithm for classification
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# convert an array of values into a timeseries of 3 previous steps matrix
def create_data(data, truth,inserted):
    dataX = []
    dataY = []
    for i in range(len(data)):
        if inserted[i] == 1:
            dataX.append(data[i])
            dataY.append(truth[i])
    return np.array(dataX),np.array(dataY)

#same results for same model, makes it deterministic
np.random.seed(1234)


#reading data
input = np.load("../Datasets/datasets_star_whiterandom/random_time_0.5.npy", allow_pickle=True)
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
X,Y= create_data(data,truth,inserted)
X = X.astype('float64')



# define the model
model = RandomForestClassifier(verbose=1,random_state=1234)
model.fit(X,list(Y))

pickle.dump(model, open('RandomForestStarWhiteRandom.sav', 'wb'))