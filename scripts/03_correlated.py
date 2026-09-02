"""Correlated inputs: uncertainty analysis with rho = 0.7 between h_u and h_l.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uq_sa_assignment.borehole_func import func
from uq_sa_assignment.distributions import (NAMES, correlation_matrix,
                                            iman_conover, sample_X)

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(exist_ok=True)

N = 10**5
SEED = 1234
RHO = 0.7

i_hu = NAMES.index('h_u')
i_hl = NAMES.index('h_l')

# --- the two samples --------------------------------------------------------
X0 = sample_X(N, SEED)                                    
Y0 = func(X0)

X7 = iman_conover(X0, correlation_matrix(RHO), seed=42)
Y7 = func(X7)

# --- statistics side by side ------------------------------------------------
def stats(Y):
    return {
        'mean': Y.mean(),
        'sd': Y.std(ddof=1),
        'var': Y.var(ddof=1),
        'MCerr': Y.std(ddof=1) / np.sqrt(N),        # error on the mean
        'median': np.median(Y),
    }

s0, s7 = stats(Y0), stats(Y7)

print(f"\n{'quantity':10} {'independent':>12} {'rho = ' + str(RHO):>12} {'change':>10}")
for k in ['mean', 'sd', 'var', 'MCerr', 'median']:
    print(f"{k:10} {s0[k]:>12.4f} {s7[k]:>12.4f} {s7[k]-s0[k]:>+10.4f}")

# --- figure: histograms and CDFs overlaid -----------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))

for Y, lab, c in [(Y0, 'independent', '#4C72B0'), (Y7, rf'$\rho$ = {RHO}', '#C44E52')]:
    axs[0].hist(Y, bins=80, density=True, histtype='step', lw=1.8,
                color=c, label=f'{lab}   sd = {Y.std(ddof=1):.2f}')
    ys = np.sort(Y)
    axs[1].step(ys, np.arange(1, N + 1) / N, where='post', lw=1.8, color=c,
                label=lab)

axs[0].set_ylabel('Density')
axs[0].set_title('Output distribution')
axs[1].set_ylabel('Cumulative probability  $F(y)$')
axs[1].set_ylim(0, 1)
axs[1].set_title('CDF')

for a in axs:
    a.set_xlabel('Flow rate  [m$^3$/yr]')
    a.legend(frameon=False)
    a.grid(alpha=0.3)
    a.spines[['top', 'right']].set_visible(False)

fig.suptitle(f'Effect of correlating $h_u$ and $h_l$   '
             f'(LHS + Iman-Conover, N = {N})')
fig.tight_layout()
fig.savefig(FIGDIR / 'correlated_uncertainty.png', dpi=150)

print(f"\nFigure written to {FIGDIR / 'correlated_uncertainty.png'}")


# Sensitivity analysis: random sampling + binning (scatter plots)
N_SA = 10**5
NBINS = 25

Xr = sample_X(N_SA, SEED, method='mc')                      # random sampling
Xr7 = iman_conover(Xr, correlation_matrix(RHO), seed=42)    # same values, repaired
Yr, Yr7 = func(Xr), func(Xr7)

def binned_index(x, y, nbins):
    """Sort by x, average y inside each of nbins equal-count slices.

    Returns the slice centres, the slice averages, and
    var(slice averages) / var(all y)  --  the first-order sensitivity index.
    """
    order = np.argsort(x)
    groups = np.array_split(order, nbins)
    centres = np.array([x[g].mean() for g in groups])
    means = np.array([y[g].mean() for g in groups])
    return centres, means, means.var() / y.var()


print(f"\nBinning sensitivity analysis   N = {N_SA}, {NBINS} bins "
      f"({N_SA // NBINS} samples per bin)")
print(f"  {'param':6} {'S_i indep':>10} {'S_i rho=0.7':>12} "
      f"{'change':>9}")

res = {}
for j, nm in enumerate(NAMES):
    c0, m0, s0i = binned_index(Xr[:, j], Yr, NBINS)
    c7, m7, s7i = binned_index(Xr7[:, j], Yr7, NBINS)
    res[nm] = (c0, m0, s0i, c7, m7, s7i)
    print(f"  {nm:6} {s0i:>10.4f}{s7i:>12.4f} "
          f"{s7i - s0i:>+9.4f}")

print(f"\n  sum of S_i:  independent {sum(res[n][2] for n in NAMES):.4f}   "
      f"correlated {sum(res[n][5] for n in NAMES):.4f}")

# --- scatter plots with the binned conditional means ------------------------
fig2, axs2 = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)

for j, (nm, ax) in enumerate(zip(NAMES, axs2.flat)):
    c0, m0, s0i, c7, m7, s7i = res[nm]
    ax.plot(Xr7[:, j], Yr7, '.', ms=1, alpha=0.08, color='#8FA8D0')
    ax.plot(c0, m0, 'o--', color='#4C72B0', lw=1.6, ms=4,
            label=f'indep  $S_i$={s0i:.3f}')
    ax.plot(c7, m7, 'o-', color='#C44E52', lw=2, ms=4,
            label=rf'$\rho$=0.7  $S_i$={s7i:.3f}')
    ax.set_xlabel(nm)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)

for ax in axs2[:, 0]:
    ax.set_ylabel('Flow rate  [m$^3$/yr]')

fig2.suptitle(f'Binning sensitivity analysis: average flow within slices of '
              f'each input   (random sampling, N = {N_SA}, {NBINS} bins)')
fig2.tight_layout()
fig2.savefig(FIGDIR / 'correlated_binning.png', dpi=150)

print(f"Figure written to {FIGDIR / 'correlated_binning.png'}")
