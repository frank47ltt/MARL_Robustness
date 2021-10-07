import matplotlib.pyplot as plt

x_axis = [.25,.50,.75,1.00]

dense_fn =          [0.06068,   .02342,  0.01632,   0.02257]
binary_lstm_fn =    [0.06832,   .02463,  0.01598,   0.03443]
random_fn =         [0.10042,   0.03841, 0.02390,   0.02177]
svm_fn =            [0.30437,   .17641,  0.10609,   0.08828]
knearest_fn =       [0.32277,   .22997,  0.20936,   0.23673]
ensemble_fn =       [0.01058,   .01455,  0.00717,   0.02025]

plt.plot(x_axis,dense_fn, 'k--',label='Binary Dense')
plt.plot(x_axis,binary_lstm_fn, 'k:',label='Binary LSTM')
plt.plot(x_axis,random_fn,'k-.', label='Random Forest')
plt.plot(x_axis,svm_fn, label='SVM Classifier')
plt.plot(x_axis,knearest_fn, label='KNearest Neighbors')
plt.plot(x_axis,ensemble_fn, 'r-',label='Proposed Ensemble Model')
plt.title("False Negative Rate For White-Box Random-Timed Attack on Cooperative Navigation Environment")
plt.xlabel("Adversarial Attack Rate")
plt.legend()
plt.xticks(x_axis)
plt.show()


plt.clf()   # clear figure



dense_fn =          [0.06226,	0.04377,	0.04292,	0.05456]
binary_lstm_fn =    [0.06729,	0.04263,	0.04769,	0.06785]
random_fn =         [0.05950,   0.047900,   .050150,    .06053]
svm_fn =            [0.337010,  .269870,    .203340,    .17617]
knearest_fn =       [0.182920,  .269870,    .148620,    .17324]
ensemble_fn =       [0.04038,   0.02558,	0.02902,	0.04178]

plt.plot(x_axis,dense_fn, 'k--',label='Binary Dense')
plt.plot(x_axis,binary_lstm_fn, 'k:',label='Binary LSTM')
plt.plot(x_axis,random_fn,'k-.', label='Random Forest')
plt.plot(x_axis,svm_fn, label='SVM Classifier')
plt.plot(x_axis,knearest_fn, label='KNearest Neighbors')
plt.plot(x_axis,ensemble_fn, 'r-',label='Proposed Ensemble Model')
plt.title("False Negative Rate For White-Box Random-Timed Attack on Physical Deception Environment")
plt.xlabel("Adversarial Attack Rate")
plt.legend()
plt.xticks(x_axis)
plt.show()


plt.clf()   # clear figure


dense_fn =          []
binary_lstm_fn =    []
ensemble_fn =       []

plt.plot(x_axis,dense_fn, 'k--',label='Binary Dense')
plt.plot(x_axis,binary_lstm_fn, 'k:',label='Binary LSTM')
plt.plot(x_axis,ensemble_fn, 'r-',label='Proposed Ensemble Model')
plt.title("False Negative Rate Against a White-Box Prediction-Based Attack on Cooperative Navigation Environment")
plt.xlabel("Adversarial Attack Rate")
plt.xticks(x_axis)
plt.legend()
plt.show()