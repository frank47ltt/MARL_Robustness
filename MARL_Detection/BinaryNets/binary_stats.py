import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

history = np.load("binary_lstm_history_nav_random.npy", allow_pickle=True).item()

model = load_model("DenseNavWhiteRandom.keras")
model.summary()


plt.plot(history['loss'], label='Training loss')
plt.plot(history['val_loss'], label='Validation loss')
plt.title("Binary Anomaly/Non-anomaly Loss")
plt.legend()
plt.show()

plt.clf()   # clear figure

plt.plot(history['acc'], label='Training acc')
plt.plot(history['val_acc'], label='Validation acc')
plt.title("Binary Anomaly/Non-anomaly Accuracy")
plt.legend()
plt.show()
