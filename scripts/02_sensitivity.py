"""Sensitivity analysis: standardized regression coefficients and Sobol' indices.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear

from uq_sa_assignment.borehole_func import func
from uq_sa_assignment.distributions import M, NAMES, sample_X

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(exist_ok=True)

N = 2**10          # same sample as the uncertainty analysis
SEED = 1234
N_SOBOL = 2**16    # Sobol' indices need far more samples than the mean does

X = sample_X(N, SEED)
Y = func(X)

# ---------------------------------------------------------------------------
# SRC: standardized regression coefficients
# ---------------------------------------------------------------------------
# standarize inputs and output
Xs = (X - X.mean(axis=0)) / X.std(axis=0)
Ys = (Y - Y.mean()) / Y.std()

res = lsq_linear(Xs, Ys, bounds=(-np.ones(M), np.ones(M)))
SRC = res.x

Ys_hat = Xs @ SRC
R2 = 1 - ((Ys - Ys_hat) ** 2).sum() / ((Ys - Ys.mean()) ** 2).sum()

print(f"SRC analysis:  R2 = {R2:.4f}")
print(f"  sum of SRC^2          = {(SRC**2).sum():.4f}")

order = np.argsort(-np.abs(SRC))
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.bar([NAMES[j] for j in order], SRC[order],
       color=['#4C72B0' if SRC[j] > 0 else '#C44E52' for j in order])
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('Input parameter')
ax.set_ylabel('SRC')
ax.set_title(f'Standardized regression coefficients   ($R^2$ = {R2:.3f})')
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(FIGDIR / 'src.png', dpi=150)

# ---------------------------------------------------------------------------
# Sobol': pick-and-freeze on two independent samples
# ---------------------------------------------------------------------------
XA = sample_X(N_SOBOL, SEED)         # first independent sample
XB = sample_X(N_SOBOL, 5678)         # second independent sample
YA = func(XA)
YB = func(XB)

VA = np.mean(YA**2) - np.mean(YA)**2

Si = np.zeros(M)                     # first-order indices
STi = np.zeros(M)                    # total-order indices

for i in range(M):
    XAB = XA.copy()
    XAB[:, i] = XB[:, i]             # A, with column i frozen from B
    XBA = XB.copy()
    XBA[:, i] = XA[:, i]             # B, with column i frozen from A

    YAB = func(XAB)
    YBA = func(XBA)

    Si[i] = (np.mean(YA * YBA) - np.mean(YA) * np.mean(YBA)) / VA
    STi[i] = np.mean(YA * (YA - YAB)) / VA

# comparison table
print(f"\nSobol' analysis   N = {N_SOBOL}, "
      f"{(2 * M + 2) * N_SOBOL} model evaluations")
print(f"  {'param':6} {'S_i':>8} {'S_Ti':>8} {'S_Ti-S_i':>9} "
      f"{'SRC^2':>8} {'S_i-SRC^2':>10}")
for j in np.argsort(-STi):
    print(f"  {NAMES[j]:6} {Si[j]:>8.4f} {STi[j]:>8.4f} {STi[j]-Si[j]:>9.4f} "
          f"{SRC[j]**2:>8.4f} {Si[j]-SRC[j]**2:>10.4f}")

print(f"\n  sum S_i  = {Si.sum():.4f}   <- additive share of the variance")
print(f"  sum S_Ti = {STi.sum():.4f}   <- must be >= 1")
print(f"  R2 (linear main effects, from SRC)    = {R2:.4f}")
print(f"  nonlinear main effects  sum S_i - R2  = {Si.sum() - R2:.4f}")

# --- comparison of the two methods ------------------------------------------
order = np.argsort(-STi)
xpos = np.arange(M)
w = 0.27

fig2, ax2 = plt.subplots(figsize=(9, 4.4))
ax2.bar(xpos - w, Si[order], w, label="Sobol' $S_i$ (first order)", color='#4C72B0')
ax2.bar(xpos, STi[order], w, label="Sobol' $S_{Ti}$ (total)", color='#8FA8D0')
ax2.bar(xpos + w, SRC[order]**2, w, label='SRC$^2$', color='#DD8452')
ax2.set_xticks(xpos)
ax2.set_xticklabels([NAMES[j] for j in order])
ax2.set_xlabel('Input parameter')
ax2.set_ylabel('Share of output variance')
ax2.set_title("Sensitivity indices: Sobol' vs SRC   "
              f"($R^2$ = {R2:.3f},  interactions = {1-Si.sum():.3f})")
ax2.legend(frameon=False)
ax2.grid(axis='y', alpha=0.3)
ax2.spines[['top', 'right']].set_visible(False)
fig2.tight_layout()
fig2.savefig(FIGDIR / 'sensitivity_comparison.png', dpi=150)

print(f"\nFigures written to {FIGDIR}")
