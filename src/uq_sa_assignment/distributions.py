"""Input distributions for the borehole function (Table 1 of the assignment).
"""

import numpy as np
from scipy.stats import lognorm, norm, qmc, uniform


def _uniform(lower, upper):
    """Uniform on [lower, upper] -- scipy wants (loc, width)."""
    return uniform(loc=lower, scale=upper - lower)


def _lognormal(mu_log, sigma_log):
    """Lognormal from the mean and sd of ln(X)."""
    return lognorm(s=sigma_log, scale=np.exp(mu_log))

# Table 1 marginals
r_w = _lognormal(0.1, 0.02)              # radius of the borehole [m]
r = _uniform(50, 150)                    # radius of influence [m]
t_u = _uniform(63070, 115600)            # transmissivity, upper aquifer [m2/yr]
h_u = _uniform(990, 1110)                # potentiometric head, upper aquifer [m]
t_l = _uniform(65, 1165)                 # transmissivity, lower aquifer [m2/yr]
h_l = _uniform(700, 820)                 # potentiometric head, lower aquifer [m]
l = _uniform(1120, 1680)                 # length of borehole [m]
k_w = norm(loc=10000, scale=1500)        # hydraulic conductivity [m/yr]

NAMES = ['r_w', 'r', 't_u', 'h_u', 't_l', 'h_l', 'l', 'k_w']
DISTS = [r_w, r, t_u, h_u, t_l, h_l, l, k_w]
M = len(DISTS)                           # number of input parameters


def sample_X(n, seed, method='lhs'):
    """Draw n scenarios -> array of shape (n, M) in real parameter units.

    method='lhs' gives a Latin hypercube design, 'mc' plain random sampling.
    """
    if method == 'lhs':
        u = qmc.LatinHypercube(d=M, seed=seed).random(n=n)
    elif method == 'mc':
        u = np.random.default_rng(seed).random((n, M))
    else:
        raise ValueError(f"unknown method {method!r}, expected 'lhs' or 'mc'")

    return np.stack([d.ppf(u[:, j]) for j, d in enumerate(DISTS)], axis=1)


def correlation_matrix(rho, pair=('h_u', 'h_l')):
    """M x M target correlation matrix: identity, plus rho between `pair`."""
    C = np.eye(M)
    i, j = NAMES.index(pair[0]), NAMES.index(pair[1])
    C[i, j] = C[j, i] = rho
    return C


def iman_conover(X, target_corr, seed=0):
    """Reorder each column of X so the sample matches `target_corr`.

    Every column keeps exactly the values it already had, so the marginals
    are preserved exactly. Only which values share a row changes.
    """
    n, m = X.shape

    # 1. a throwaway template with the correlation we want.
    L = np.linalg.cholesky(target_corr)
    Z = np.random.default_rng(seed).standard_normal((n, m))
    Zc = Z @ L.T

    # 2. the template's rank pattern.
    ranks = np.argsort(np.argsort(Zc, axis=0), axis=0)

    # 3. deal our real values out following that pattern
    X_sorted = np.sort(X, axis=0)
    out = np.zeros_like(X)
    for j in range(m):
        out[:, j] = X_sorted[ranks[:, j], j]
    return out
