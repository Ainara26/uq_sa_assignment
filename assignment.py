import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, uniform, norm, qmc
from src.uq_sa_assignment.borehole_func import func

# Define parameter distribution
m = 8 # total number of parameters
mu, sigma = 0.1, 0.02                    # Table 1
cv2 = (sigma / mu) ** 2
r_w = lognorm(s=np.sqrt(np.log(1 + cv2)), scale=mu / np.sqrt(1 + cv2))
r = uniform(loc=50, scale=150-50)
t_u = uniform(loc=63070, scale=115600-63070)
h_u = uniform(loc=990, scale=1110-990)
t_l = uniform(loc=65, scale=1165-65)
h_l = uniform(loc=700, scale=820-700)
l = uniform(loc=1120, scale=1680-1120)
k_w = norm(loc=10000, scale=1500)

N = 2**10  # Monte Carlo samples
seed = 1234
sampler = qmc.LatinHypercube(d=m, seed=seed)
sample = sampler.random(n=N)

X_rw = r_w.ppf(sample[:, 0])
X_r = r.ppf(sample[:, 1])
X_tu = t_u.ppf(sample[:, 2])
X_hu = h_u.ppf(sample[:, 3])
X_tl = t_l.ppf(sample[:, 4])
X_hl = h_l.ppf(sample[:, 5])
X_l = l.ppf(sample[:, 6])
X_kw = k_w.ppf(sample[:, 7])

X = np.stack([X_rw, X_r, X_tu, X_hu, X_tl, X_hl, X_l, X_kw], axis=1)
Y = func(X)

print("Mean of f(x):",Y.mean())
print("Standard deviation of f(x)",Y.std(ddof=1))
print("Variance of f(x)",Y.var(ddof=1))    

plt.hist(Y, bins='auto', density=True)
plt.axvline(x=Y.mean())
plt.xlabel('Flow rate (m3/yr)')
plt.ylabel('Density')
plt.savefig('histogram.png', dpi=150)
