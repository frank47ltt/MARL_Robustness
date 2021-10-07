import matplotlib.pyplot as plt
import numpy as np

#whitebox random, 3 bars per technique corresponds to 3 environments
#0 = SC,1=CN,2=PD
ensemble = [94.19,97.84,96.58]
fgsm = [57.9,82.1,80.7]

ensemble = np.round(np.asarray(ensemble))
fgsm = np.round(np.asarray(fgsm))

starcraft = [ensemble[0],fgsm[0]]
nav = [ensemble[1],fgsm[1]]
simp = [ensemble[2],fgsm[2]]

N = 2
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

rects1 = ax.bar(ind, starcraft, width, color='r')
rects2 = ax.bar(ind+width, nav, width, color='g')
rects3 = ax.bar(ind+width*2, simp, width, color='b')
plt.ylim(55, 100)
ax.set_ylabel('Recall (%)')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Ensemble','Attacked\nEnsemble') )
ax.legend( (rects1[0], rects2[0], rects3[0]), ('Starcraft', 'Navigation', 'Deception') )
plt.gca().get_xticklabels()[1].set_color('red')

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.001*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)



plt.show()