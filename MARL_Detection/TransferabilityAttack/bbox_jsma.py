import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from cleverhans.attacks import SaliencyMapMethod


X = np.load('fgsm_data_nav/query_navX.npy')
Y = np.load('fgsm_data_nav/query_navY.npy')
testX = X[200000:]
testY = Y[200000:]

num_classes = 2
num_iter = 200 #testX.shape[0]

#The attack requires the model to ouput the logits
model = load_model('ReplicaNavEnsembleLogits.keras')
logits_model = tf.keras.Model(model.input,model.layers[-1].output)
jsma = SaliencyMapMethod(model)



for j in range(1,21):
    successful_indices = []
    attacked_X = []
    results = []
    feature_perturbed = np.zeros((num_iter,testX.shape[1],testX.shape[2]))
    epsilon = j*.05
    jsma_params = {"theta": 1.0,"gamma": epsilon,"clip_min": 0.0,"clip_max": 1.0,"y_target": None}


    for i in range(num_iter):
        if testY[i] == 1:
            vector = testX[i:i+1]
            tf.convert_to_tensor(vector.astype('float'))
            original_label = np.reshape(testY[i], (1,)).astype('int64')  # Give label proper shape and type for cleverhans

            target = [1,0]   #one hot encoding for target class. target class is 0
            jsma_params["y_target"] = target
            adv_x = jsma.generate_np(vector, **jsma_params)

            #only actually perturb the first agent's observations, indices [0:18]
            #adv_x = np.concatenate((adv_x[:,:,:18],vector[:,:,18:]),axis=2)
            adv_x_pred = np.argmax(model.predict(adv_x))

            # Compute number of modified features
            adv_x_reshape = adv_x.reshape(-1)
            test_in_reshape = testX[i].reshape(-1)
            were_changed = np.argwhere(adv_x_reshape != test_in_reshape)
            print(were_changed)

            if adv_x_pred == 0:
                results.append(1)
                successful_indices.append(i)
                attacked_X.append(adv_x[0])
            else:
                results.append(0)
        if i%1000 == 0:
            print("Iteration ",i)
    results = np.asarray(results)
    attacked_X = np.array(attacked_X)
    print("Success rate of fooling the replica model with epsilon", epsilon,": " + str(np.sum(results)/results.shape[0]))

    newY = []
    trueActions = []
    j = 0
    for i in range(len(results)):
        if results[i] == 1:
            trueActions.append(testX[successful_indices[j],:,-3:])
            newY.append(testY[successful_indices[j]])
            j +=1
    trueActions = np.asarray(trueActions)
    newY = np.asarray(newY)

    np.save("fgsm_testX" + str(np.round(epsilon,2) * 100),attacked_X)
    print(trueActions.shape)
    print(np.unique(trueActions))
    np.save("fgsm_testY"+ str(np.round(epsilon,2) * 100),newY)
    np.save("fgsm_trueActions"+ str(np.round(epsilon,2) * 100),trueActions)