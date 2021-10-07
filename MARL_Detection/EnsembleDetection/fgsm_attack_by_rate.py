import matplotlib.pyplot as plt
import numpy as np

x_axis = [.25,.50,.75,1.00]

nav = [4.22,1.45,0.96,2.02,]
star= [2.24,3.69,7.38,9.94,]
simp= [4.04,2.56,2.9,4.18,]
nav_fgsm = [45.63,17.75,4.83,3.45]
star_fgsm = [33.11,38.42,43.61,53.39]
simp_fgsm =[28.28,21.09,14.83,12.91]

plt.plot(x_axis,nav, 'g:',label='Nav')
plt.plot(x_axis,nav_fgsm, 'g',label='Nav FGSM')
plt.plot(x_axis,simp, 'r:',label='Phys')
plt.plot(x_axis,simp_fgsm, 'r',label='Phys FGSM')
plt.plot(x_axis,star, 'b:',label='Star')
plt.plot(x_axis,star_fgsm, 'b',label='Star FGSM')

plt.title("Percentage of Undetected Positives \n Before and After FGSM")
plt.xlabel("Adversarial Attack Rate")
plt.ylabel("False Negative Rate (%)")
plt.legend()
plt.xticks(x_axis)
plt.show()


plt.clf()   # clear figure

nav = [4.22,1.45,0.96,2.02,]
star= [2.24,3.69,7.38,9.94,]
simp= [4.04,2.56,2.9,4.18,]
nav_fgsm = [45.63,17.75,4.83,3.45]
star_fgsm = [33.11,38.42,43.61,53.39]
simp_fgsm =[28.28,21.09,14.83,12.91]

nav = 100 - np.asarray(nav)
nav_fgsm = 100 - np.asarray(nav_fgsm)
star = 100 - np.asarray(star)
star_fgsm = 100 - np.asarray(star_fgsm)
simp = 100 - np.asarray(simp)
simp_fgsm = 100 - np.asarray(simp_fgsm)

plt.plot(x_axis,nav, 'g:',label='Navigation')
plt.plot(x_axis,nav_fgsm, 'g',label='Nav. FGSM')
plt.plot(x_axis,simp, 'r:',label='Deception')
plt.plot(x_axis,simp_fgsm, 'r',label='Dec. FGSM')
plt.plot(x_axis,star, 'b:',label='Starcraft')
plt.plot(x_axis,star_fgsm, 'b',label='Star. FGSM')
plt.xlabel("Adversarial Attack Rate")
plt.ylabel("Recall (%)")
plt.legend()
plt.xticks(x_axis)
plt.show()


plt.clf()   # clear figure
