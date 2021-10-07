import matplotlib.pyplot as plt
import numpy as np

x_axis = [.25,.50,.75,1.00]

#WHITEBOX RANDOM STATS HERE
dense_fn =          [0.06068,   .02342,  0.01632,   0.02257]
random_fn =         [0.10042,   0.03841, 0.02390,   0.02177]
svm_fn =            [0.30437,   .17641,  0.10609,   0.08828]
knearest_fn =       [0.32277,   .22997,  0.20936,   0.23673]
ensemble_fn =       [0.01058,   .01455,  0.00717,   0.02025]

plt.plot(x_axis,dense_fn, 'k--',label='Binary Dense')
plt.plot(x_axis,random_fn,'k-.', label='Random Forest')
plt.plot(x_axis,svm_fn, 'k-',label='SVM Classifier')
plt.plot(x_axis,knearest_fn, 'k:',label='KNearest Neighbors')
plt.plot(x_axis,ensemble_fn, 'r-',label='Proposed Ensemble Model')
plt.title("Cooperative Navigation Environment")
plt.xlabel("Adversarial Attack Rate")
plt.legend()
plt.xticks(x_axis)
plt.ylim(0, .35)
plt.show()


plt.clf()   # clear figure



dense_fn =          [0.06226,	0.04377,	0.04292,	0.05456]
random_fn =         [0.05950,   0.047900,   .050150,    .06053]
svm_fn =            [0.337010,  .269870,    .203340,    .17617]
knearest_fn =       [0.182920,  .269870,    .148620,    .17324]
ensemble_fn =       [0.04038,   0.02558,	0.02902,	0.04178]

plt.plot(x_axis,dense_fn, 'k--',label='Binary Dense')
plt.plot(x_axis,random_fn,'k-.', label='Random Forest')
plt.plot(x_axis,svm_fn, 'k-',label='SVM Classifier')
plt.plot(x_axis,knearest_fn, 'k:',label='KNearest Neighbors')
plt.plot(x_axis,ensemble_fn, 'r-',label='Proposed Ensemble Model')
plt.title("Physical Deception Environment")
plt.xlabel("Adversarial Attack Rate")
plt.legend()
plt.xticks(x_axis)
plt.ylim(0, .35)
plt.show()


plt.clf()   # clear figure



dense_fn =          [3.57,6.81,15.94,25.17]
dense_fn = np.asarray(dense_fn)
dense_fn = dense_fn/100
random_fn =         [0.04568,0.09459,0.20242,0.29765]
svm_fn =            [0.12922,0.17248,0.23345,0.16642]
knearest_fn =       [0.05506,0.09782,0.20527,0.29842]
ensemble_fn =       [2.24,3.69,7.38,9.94]
ensemble_fn = np.asarray(ensemble_fn)
ensemble_fn = ensemble_fn/100

plt.plot(x_axis,dense_fn, 'k--',label='Binary Dense')
plt.plot(x_axis,random_fn,'k-.', label='Random Forest')
plt.plot(x_axis,svm_fn, 'k-', label='SVM Classifier')
plt.plot(x_axis,knearest_fn, 'k:', label='KNearest Neighbors')
plt.plot(x_axis,ensemble_fn, 'r-',label='Proposed Ensemble Model')
plt.title("StarCraft II Environment")
plt.xlabel("Adversarial Attack Rate")
plt.legend()
plt.xticks(x_axis)
plt.ylim(0, .30)
plt.show()
