import matplotlib.pyplot as plt
import numpy as np

x_axis = np.asarray(range(1,21)) * .05
print(x_axis)


#fgsm undetected positives
nav =          np.asarray([0.166,0.313,0.429,0.499,0.543,0.572,0.586,0.583,0.569,0.551,0.529,0.511,0.484,0.465,0.446,0.421,0.401,0.385,0.374,0.358])
simp =         np.asarray([0.056,0.113,0.164,0.216,0.274,0.315,0.337,0.361,0.374,0.387,0.392,0.391,0.394,0.393,0.393,0.393,0.393,0.396,0.392,0.389])
star =         np.asarray([0.17,0.226,0.205,0.212,0.227,0.24,0.261,0.281,0.302,0.318,0.331,0.336,0.34,0.341,0.34,0.337,0.339,0.329,0.325,0.316])

best_nav = np.argmax(nav)
best_simp = np.argmax(simp)
best_star = np.argmax(star)


plt.plot(x_axis,nav,label='Cooperative Navigation')
plt.plot(x_axis,simp, label='Physical Deception')
plt.plot(x_axis,star,label='Starcraft II')
plt.plot(best_nav*0.05+0.05,nav[best_nav], "ro",label="Most Effective FGSM Epsilon")
plt.plot(best_simp*0.05+0.05,simp[best_simp], "ro")
plt.plot(best_star*0.05+0.05,star[best_star], "ro")
plt.title("Undetected Positive Rate using FGSM to \nFool Detection Ensemble Model")
plt.xlabel("FGSM Epsilon")
plt.legend()
plt.xticks(np.asarray(range(1,11))*.10)
#plt.ylim(0, .35)
plt.show()


plt.clf()

#fgsm undetected positives
nav =          1 - np.asarray([0.166,0.313,0.429,0.499,0.543,0.572,0.586,0.583,0.569,0.551,0.529,0.511,0.484,0.465,0.446,0.421,0.401,0.385,0.374,0.358])
simp =         1 - np.asarray([0.056,0.113,0.164,0.216,0.274,0.315,0.337,0.361,0.374,0.387,0.392,0.391,0.394,0.393,0.393,0.393,0.393,0.396,0.392,0.389])
star =         1 - np.asarray([0.17,0.226,0.205,0.212,0.227,0.24,0.261,0.281,0.302,0.318,0.331,0.336,0.34,0.341,0.34,0.337,0.339,0.329,0.325,0.316])

best_nav = np.argmin(nav)
best_simp = np.argmin(simp)
best_star = np.argmin(star)


plt.plot(x_axis,nav,label='Cooperative Navigation')
plt.plot(x_axis,simp, label='Physical Deception')
plt.plot(x_axis,star,label='Starcraft II')
plt.plot(best_nav*0.05+0.05,nav[best_nav], "ro",label="Most Effective FGSM Epsilon")
plt.plot(best_simp*0.05+0.05,simp[best_simp], "ro")
plt.plot(best_star*0.05+0.05,star[best_star], "ro")
plt.title("Recall using FGSM to \nFool Detection Ensemble Model")
plt.xlabel("FGSM Epsilon")
plt.legend()
plt.xticks(np.asarray(range(1,11))*.10)
#plt.ylim(0, .35)
plt.show()


plt.clf()
