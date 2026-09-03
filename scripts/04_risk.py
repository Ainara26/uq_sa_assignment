from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

from uq_sa_assignment.borehole_func import func
from uq_sa_assignment.distributions import (DISTS, NAMES, correlation_matrix,
                                            iman_conover, sample_from,
                                            sample_X)

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(exist_ok=True)

N = 2**10          # Monte Carlo samples
SEED = 1234

X = sample_X(N, SEED)
Y = func(X)

threshold = 9500

probability = (Y>threshold).mean() 
print(f"probability of exceeding threshold (%):", probability*100)

def exceed(Y):
    """Exceedance probability and its binomial standard error."""
    p = (Y > threshold).mean()
    return p, np.sqrt(p * (1 - p) / len(Y))


dists_mit = list(DISTS)                                    # a copy of Table 1
dists_mit[NAMES.index('k_w')] = norm(loc=10000, scale=750)  # sd 1500 -> 750

X_mit = sample_from(dists_mit, N, SEED)
Y_mit = func(X_mit)

p_base, se_base = exceed(Y)
p_mit, se_mit = exceed(Y_mit)

print(f"\n{'case':22} {'mean':>10} {'sd':>10} {'P(exceed)':>11} {'se':>8}")
print(f"{'(a) base':22} {Y.mean():>10.1f} {Y.std(ddof=1):>10.1f} "
      f"{p_base:>11.4f} {se_base:>8.4f}")
print(f"{'(b) K_w sd halved':22} {Y_mit.mean():>10.1f} {Y_mit.std(ddof=1):>10.1f} "
      f"{p_mit:>11.4f} {se_mit:>8.4f}")
print(f"\n  output sd changed by {100*(Y_mit.std(ddof=1)/Y.std(ddof=1)-1):+.1f} %")
print(f"  risk fell by {100*(p_base-p_mit):.2f} percentage points "
      f"-> {p_base/p_mit:.2f}x lower")
