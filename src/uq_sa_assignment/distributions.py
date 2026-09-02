"""Input distributions for the borehole function (Table 1 of the assignment).
"""

import numpy as np
from scipy.stats import lognorm, norm, qmc, uniform


def _uniform(lower, upper):
    """Uniform on [lower, upper] -- scipy wants (loc, width)."""
    return uniform(loc=lower, scale=upper - lower)


def _lognormal(mean, sd):
    """Lognormal from the MEAN and SD of the variable itself, not of its log."""
    cv2 = (sd / mean) ** 2
    return lognorm(s=np.sqrt(np.log(1 + cv2)), scale=mean / np.sqrt(1 + cv2))


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


def sample_X_correlated(n, seed, rho):
    """Like sample_X, but with h_u and h_l positively correlated.
    """
    rng = np.random.default_rng(seed)

    # 1. eight columns of bell-curve numbers, not uniform ones: blending
    z = rng.standard_normal((n, M))

    # 2. rebuild the h_l column as a blend of its partner and itself.
    i_hu = NAMES.index('h_u')
    i_hl = NAMES.index('h_l')
    z[:, i_hl] = rho * z[:, i_hu] + np.sqrt(1 - rho**2) * z[:, i_hl]

    # 3. back to numbers between 0 and 1. norm.cdf preserves order, so the
    u = norm.cdf(z)

    # 4. same final step as sample_X
    return np.stack([d.ppf(u[:, j]) for j, d in enumerate(DISTS)], axis=1)
