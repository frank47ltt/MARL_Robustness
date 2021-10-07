import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

history1 = np.load("replica0_history.npy", allow_pickle=True).item()
history2 = np.load("replica0_history.npy", allow_pickle=True).item()

model = load_model("ReplicaModel0.keras")
model.summary()

plt.clf()   # clear figure

plt.plot(history1['val_acc'], label='Replica acc')
plt.plot(history2['val_acc'], label='Patient Replica acc')
plt.legend()
plt.show()

print(history1['val_acc'][-1])
print(history2['val_acc'][-1])