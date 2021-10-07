import matplotlib.pyplot as plt
import numpy as np

#whitebox random, 3 bars per technique corresponds to 3 environments
#0 = SC,1=CN,2=PD
ensemble = [94.19,97.84,96.58]
dense = [87.13,96.92,94.91]
lstm = [85.66,96.42,94.36]
forest = [83.99,95.39,94.55]
svm = [82.46,83.12,75.34]
knearest = [83.59,75.03,80.63]

ensemble = np.round(np.asarray(ensemble))

starcraft = [ensemble[0],dense[0],lstm[0],forest[0],svm[0],knearest[0]]
nav = [ensemble[1],dense[1],lstm[1],forest[1],svm[1],knearest[1]]
simp = [ensemble[2],dense[2],lstm[2],forest[2],svm[2],knearest[2]]

N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

rects1 = ax.bar(ind, starcraft, width, color='r')
rects2 = ax.bar(ind+width, nav, width, color='g')
rects3 = ax.bar(ind+width*2, simp, width, color='b')
plt.ylim(74, 100)
ax.set_ylabel('Recall (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Ensemble','Dense', 'LSTM', 'Random\nForest','SVM','Knearest') )
ax.legend( (rects1[0], rects2[0], rects3[0]), ('Starcraft', 'Navigation', 'Deception') )
plt.gca().get_xticklabels()[0].set_color('red')

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.001*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)



plt.show()