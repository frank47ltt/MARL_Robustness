
import numpy as np
from scipy.special import softmax

def kl_divergence(p, q):
    epsilon = 0.0000001
    p = p + epsilon
    q = q + epsilon
    ans = sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))
    return ans


if __name__ == "__main__":
    a = np.array([0,1,0,0,0,0,0,0,0],dtype= np.float32)
    b = np.array([1,0,0,0,0,0,0,0,0],dtype=np.float32)
    c = np.random.uniform(low=-3, high=3, size=(9,))
    print(c)
    d = np.random.uniform(low=-3, high=3, size=(9,))
    c_max = np.max(c)
    c_min = np.min(c)
    d_max = np.max(d)
    d_min = np.min(d)
    c = (c - c_min)/(c_max - c_min)
    d = (d - d_min)/(d_max - d_min)
    print(c)
    c = softmax(c)
    print(c)

    print(kl_divergence(c,d))