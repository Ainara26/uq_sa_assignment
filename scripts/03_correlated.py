"""Correlated inputs: uncertainty analysis with rho = 0.7 between h_u and h_l.

Both cases use the same sampler and the same seed, so the only thing that
differs between them is the correlation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uq_sa_assignment.borehole_func import func
from uq_sa_assignment.distributions import NAMES, sample_X_correlated

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(exist_ok=True)

N = 10**5          
SEED = 1234
RHO = 0.7

i_hu = NAMES.index('h_u')
i_hl = NAMES.index('h_l')

# --- the two samples --------------------------------------------------------
X0 = sample_X_correlated(N, SEED, rho=0.0)      # independent control
Y0 = func(X0)

X7 = sample_X_correlated(N, SEED, rho=RHO)      # correlated case
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

print(f"\n{'quantity':10} {'rho = 0':>12} {'rho = ' + str(RHO):>12} {'change':>10}")
for k in ['mean', 'sd', 'var', 'MCerr', 'median']:
    print(f"{k:10} {s0[k]:>12.4f} {s7[k]:>12.4f} {s7[k]-s0[k]:>+10.4f}")

# --- figure: histograms and CDFs overlaid -----------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))

for Y, rho, c in [(Y0, 0.0, '#4C72B0'), (Y7, RHO, '#C44E52')]:
    axs[0].hist(Y, bins=80, density=True, histtype='step', lw=1.8,
                color=c, label=rf'$\rho$ = {rho}   sd = {Y.std(ddof=1):.2f}')
    ys = np.sort(Y)
    axs[1].step(ys, np.arange(1, N + 1) / N, where='post', lw=1.8, color=c,
                label=rf'$\rho$ = {rho}')

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
             f'(random sampling, N = {N})')
fig.tight_layout()
fig.savefig(FIGDIR / 'correlated_uncertainty.png', dpi=150)

print(f"\nFigure written to {FIGDIR / 'correlated_uncertainty.png'}")
