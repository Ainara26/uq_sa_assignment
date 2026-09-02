"""Uncertainty analysis: Monte Carlo with Latin hypercube sampling.

Reports mean, standard deviation, variance and Monte Carlo error, and produces
the histogram, CDF and convergence figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uq_sa_assignment.borehole_func import func
from uq_sa_assignment.distributions import sample_X

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(exist_ok=True)

N = 2**10          # Monte Carlo samples
SEED = 1234

X = sample_X(N, SEED)
Y = func(X)

# --- statistics -------------------------------------------------------------
MCerr = Y.std(ddof=1) / np.sqrt(N)
print(f"Mean of f(x):               {Y.mean():.4f}")
print(f"Standard deviation of f(x): {Y.std(ddof=1):.4f}")
print(f"Variance of f(x):           {Y.var(ddof=1):.4f}")
print(f"Monte Carlo error:          {MCerr:.4f}")

# --- histogram --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(Y, bins='auto', color='#4C72B0', edgecolor='white', linewidth=0.5)
std = Y.std(ddof=1)
ax.axvspan(Y.mean() - std, Y.mean() + std, color='grey', alpha=0.15, zorder=0,
           label=rf'mean $\pm$ 1$\sigma$: {Y.mean()-std:.1f} - {Y.mean()+std:.1f}')
ax.axvline(Y.mean(), color='#C44E52', lw=2, label=f'mean = {Y.mean():.1f}')
ax.set_xlabel('Flow rate  [m$^3$/yr]')
ax.set_ylabel('Frequency')
ax.set_title(f'Borehole output distribution - LHS Monte Carlo, N = {N}')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(FIGDIR / 'histogram.png', dpi=150)

# --- CDF: sort the samples, i-th smallest sits at height i/N ----------------
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
fig2.savefig(FIGDIR / 'cdf.png', dpi=150)

# --- M/S analysis at many  sample sizes -------------------------------------
Ns = 2 ** np.arange(4, 16)          # 16, 32, ... 32768
M_conv = np.zeros(len(Ns))          # mean at each sample size
S_conv = np.zeros(len(Ns))          # std dev at each sample size
E_conv = np.zeros(len(Ns))          # Monte Carlo error at each sample size

for i, n in enumerate(Ns):
    y = func(sample_X(n, SEED))     
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
fig3.savefig(FIGDIR / 'convergence.png', dpi=150)

print(f"\nFigures written to {FIGDIR}")
