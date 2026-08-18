import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, uniform, norm, qmc
from scipy.optimize import lsq_linear
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

# Monte Carlo sampling
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
MCerr = Y.std(ddof=1)/np.sqrt(N)
print("Monte Carlo Error:", MCerr)

# Histogram of Y density
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(Y, bins='auto', color='#4C72B0', edgecolor='white', linewidth=0.5)
std = Y.std(ddof=1)
ax.axvspan(Y.mean() - std, Y.mean() + std, color='grey', alpha=0.15, zorder=0,
           label=rf'mean $\pm$ 1$\sigma$: {Y.mean()-std:.1f} - {Y.mean()+std:.1f}')
ax.axvline(Y.mean(), color='#C44E52', lw=2,
           label=f'mean = {Y.mean():.1f}')
ax.set_xlabel('Flow rate  [m$^3$/yr]')
ax.set_ylabel('Frequency')
ax.set_title(f'Borehole output distribution - LHS Monte Carlo, N = {N}')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig('histogram.png', dpi=150)

# CDF of Y: sort the samples, i-th smallest sits at height i/N
ys = np.sort(Y)
p = np.arange(1, N + 1) / N

fig2, ax2 = plt.subplots(figsize=(7.5, 4.2))
ax2.step(ys, p, where='post', color='#4C72B0', lw=1.8)

# how to read a percentile off the curve
for q, c in [(0.05, '#937860'), (0.50, '#DD8452'), (0.95, '#937860')]:
    yq = np.percentile(Y, 100 * q)
    ax2.hlines(q, ys[0], yq, color=c, ls='--', lw=1)
    ax2.vlines(yq, 0, q, color=c, ls='--', lw=1)
    ax2.annotate(f'P{int(q * 100)} = {yq:.1f}', xy=(yq, q),
                 xytext=(7, 5 if q < 0.1 else -13), textcoords='offset points',
                 fontsize=9, color=c)

ax2.set_xlim(ys[0], ys[-1])
ax2.set_ylim(0, 1)
ax2.set_xlabel('Flow rate  [m$^3$/yr]')
ax2.set_ylabel('Cumulative probability  $F(y)$')
ax2.set_title(f'CDF of the borehole output - LHS Monte Carlo, N = {N}')
ax2.grid(alpha=0.3)
ax2.spines[['top', 'right']].set_visible(False)
fig2.tight_layout()
fig2.savefig('cdf.png', dpi=150)

# M/S analysis  at many sample sizes
def run_mc(n, seed):
    """One Monte Carlo run of n LHS scenarios -> n flow rates."""
    u = qmc.LatinHypercube(d=m, seed=seed).random(n=n)
    Xn = np.stack([r_w.ppf(u[:, 0]), r.ppf(u[:, 1]),
                   t_u.ppf(u[:, 2]), h_u.ppf(u[:, 3]),
                   t_l.ppf(u[:, 4]), h_l.ppf(u[:, 5]),
                   l.ppf(u[:, 6]), k_w.ppf(u[:, 7])], axis=1)
    return func(Xn)

Ns = 2 ** np.arange(4, 16)          # 16, 32, ... 32768
M_conv = np.zeros(len(Ns))          # mean at each sample size
S_conv = np.zeros(len(Ns))          # std dev at each sample size
E_conv = np.zeros(len(Ns))          # Monte Carlo error at each sample size

for i, n in enumerate(Ns):
    y = run_mc(n, seed=seed)
    M_conv[i] = y.mean()
    S_conv[i] = y.std(ddof=1)
    E_conv[i] = S_conv[i] / np.sqrt(n)

fig3, axs = plt.subplots(1, 2, figsize=(14, 4))
axs[0].errorbar(Ns, M_conv, yerr=E_conv, fmt='o-', color='#4C72B0',
                capsize=3, lw=1.5, label=r'M $\pm$ MCerr')
axs[0].set_ylabel('Mean of $y$  [m$^3$/yr]')
axs[0].set_title('Mean vs sample size')

axs[1].plot(Ns, S_conv, 'o-', color='#55A868')
axs[1].axhline(S_conv[-1], color='grey', ls='--', lw=1,
               label=f'value at N = {Ns[-1]}: {S_conv[-1]:.2f}')
axs[1].set_ylabel('Std. dev. of $y$  [m$^3$/yr]')
axs[1].set_title('Standard deviation vs sample size')

for a in axs:
    a.set_xscale('log', base=2)
    a.set_xlabel('N  (sample size)')
    a.grid(alpha=0.3, which='both')
    a.legend(frameon=False, fontsize=9)
    a.spines[['top', 'right']].set_visible(False)

fig3.tight_layout()
fig3.savefig('convergence.png', dpi=150)

# SRC Method Analysis
# names of the X columns
names = ['r_w', 'r', 't_u', 'h_u', 't_l', 'h_l', 'l', 'k_w']

# Standardize inputs and output: measure everything in its own std deviations.
# After this both have mean 0, so the regression needs no intercept, and the
# fitted coefficients ARE the standardized regression coefficients.
Xs = (X - X.mean(axis=0)) / X.std(axis=0)
Ys = (Y - Y.mean()) / Y.std()

# Bounded least squares. An SRC must lie in [-1, 1], so those are the bounds.
res = lsq_linear(Xs, Ys, bounds=(-np.ones(m), np.ones(m)))
SRC = res.x

# R^2 = fraction of Var(Y) reproduced by the linear surrogate
Ys_hat = Xs @ SRC
R2 = 1 - ((Ys - Ys_hat) ** 2).sum() / ((Ys - Ys.mean()) ** 2).sum()
print(f"\nSRC analysis:  R2 = {R2:.4f}")

# SRC plot
order = np.argsort(-np.abs(SRC))
fig4, ax4 = plt.subplots(figsize=(7.5, 4.2))
ax4.bar([names[j] for j in order], SRC[order],
        color=['#4C72B0' if SRC[j] > 0 else '#C44E52' for j in order])
ax4.axhline(0, color='black', lw=0.8)
ax4.set_xlabel('Input parameter')
ax4.set_ylabel('SRC')
ax4.set_title(f'Standardized regression coefficients   ($R^2$ = {R2:.3f})')
ax4.grid(axis='y', alpha=0.3)
ax4.spines[['top', 'right']].set_visible(False)
fig4.tight_layout()
fig4.savefig('src.png', dpi=150)
