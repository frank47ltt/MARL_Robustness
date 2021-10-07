import cleverhans
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from cleverhans.future.tf2.attacks import fast_gradient_method


X = np.load('queryX.npy')
Y = np.load('queryY.npy')
testX = X[200000:]
testY = Y[200000:]

num_classes = 2
num_iter = testX.shape[0]

#The attack requires the model to ouput the logits
model = load_model('ReplicaEnsembleLogits.keras')
logits_model = tf.keras.Model(model.input,model.layers[-1].output)



for j in range(20,21):
    successful_indices = []
    attacked_X = []
    results = []
    epsilon = .05 * j
    for i in range(num_iter):
        if testY[i] == 1:
            vector = testX[i:i+1]
            tf.convert_to_tensor(vector.astype('float'))
            original_label = np.reshape(testY[i], (1,)).astype('int64')  # Give label proper shape and type for cleverhans

            adv_example_untargeted = fast_gradient_method(logits_model, vector, epsilon, np.inf, targeted=False)

            #only actually perturb the first agent's observations, indices [0:18]
            adv_example_untargeted = np.concatenate((adv_example_untargeted[:,:,:18],vector[:,:,18:]),axis=2)
            adv_example_untargeted_pred = np.argmax(model.predict(adv_example_untargeted))

            if adv_example_untargeted_pred == 0:
                results.append(1)
                successful_indices.append(i)
                attacked_X.append(adv_example_untargeted[0])
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

    np.save("fgsm_testX" + str(epsilon*100),attacked_X)
    print(trueActions.shape)
    print(np.unique(trueActions))
    np.save("fgsm_testY"+ str(epsilon*100),newY)
    np.save("fgsm_trueActions"+ str(epsilon*100),trueActions)