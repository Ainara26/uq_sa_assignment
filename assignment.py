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

# Monte Carlo Sampling
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

# Statistical parameters
print("Mean of f(x):",Y.mean())
print("Standard deviation of f(x)",Y.std(ddof=1))
print("Variance of f(x)",Y.var(ddof=1))    

# Histogram of Y density
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(Y, bins='auto', density=True,
        color='#4C72B0', edgecolor='white', linewidth=0.5)
std = Y.std(ddof=1)
ax.axvspan(Y.mean() - std, Y.mean() + std, color='grey', alpha=0.15, zorder=0,
           label=rf'mean $\pm$ 1$\sigma$: {Y.mean()-std:.1f} - {Y.mean()+std:.1f}')
ax.axvline(Y.mean(), color='#C44E52', lw=2,
           label=f'mean = {Y.mean():.1f}')
ax.set_xlabel('Flow rate  [m$^3$/yr]')
ax.set_ylabel('Density')
ax.set_title(f'Borehole output distribution - LHS Monte Carlo, N = {N}')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig('histogram.png', dpi=150)
