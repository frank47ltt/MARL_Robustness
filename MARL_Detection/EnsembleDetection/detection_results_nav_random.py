import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

x_axis = [.25,.50,.75,1.00]

dense_fn = [0.06068, .02342, 0.01632, 0.02257]
binary_lstm_fn = [0.06832,.02463,0.01598,0.03443]
random_fn = [0.10042,0.03841,0.02390,0.02177]
svm_fn = [0.30437,.17641,0.10609,0.08828]
knearest_fn = [0.32277,.22997,0.20936,0.23673]
ensemble_fn = [0.01058,.01455,0.00717,0.02025]

dense_tp = [0.93932,.97658,0.98368,0.97743]
binary_lstm_tp = [0.93168,.97537,0.98402,0.96557]
random_tp = [0.89958,.96159,0.97610,0.97823]
svm_tp = [0.69563,.82359,0.89391,0.91172]
knearest_tp = [0.67723,.77003,0.79064,0.76327]
ensemble_tp = [0.957793739,.98531,0.984121434,0.979754545]


plt.plot(x_axis,dense_tp, 'k--',label='Binary Dense',)
plt.plot(x_axis,binary_lstm_tp,'k:', label='Binary LSTM')
plt.plot(x_axis,random_tp,'k-.', label='Random Forest')
plt.plot(x_axis,svm_tp, label='SVM Classifier')
plt.plot(x_axis,knearest_tp, label='KNearest Neighbors')
plt.plot(x_axis,ensemble_tp, 'r-',label='Proposed Ensemble Model')
plt.title("Recall (aka True Positive Rate)")
plt.xlabel("Attack Rate")
plt.legend()
plt.xticks(x_axis)
plt.show()

plt.clf()   # clear figure

plt.plot(x_axis,dense_fn, 'k--',label='Binary Dense')
plt.plot(x_axis,binary_lstm_fn, 'k:',label='Binary LSTM')
plt.plot(x_axis,random_fn,'k-.', label='Random Forest')
plt.plot(x_axis,svm_fn, label='SVM Classifier')
plt.plot(x_axis,knearest_fn, label='KNearest Neighbors')
plt.plot(x_axis,ensemble_fn, 'r-',label='Proposed Ensemble Model')
plt.title("False Negative Rate")
plt.xlabel("Percent of Adversarial Datapoints (%)")
plt.legend()
plt.xticks(x_axis)
plt.show()


plt.clf()   # clear figure

plt.plot(x_axis,dense_fn, 'k--',label='Binary Dense')
plt.plot(x_axis,binary_lstm_fn, 'k:',label='Binary LSTM')
plt.plot(x_axis,ensemble_fn, 'r-',label='Proposed Ensemble Model')
plt.title("False Negative Rate of Deep Learning Models")
plt.xlabel("Percent of Adversarial Datapoints (%)")
plt.xticks(x_axis)
plt.legend()
plt.show()