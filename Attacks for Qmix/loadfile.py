import numpy as np
#from numpy import 
dict_load = np.load('data_adv.npy',allow_pickle=True)
print("Adversarial Action =",dict_load.item()['adv_action'])
