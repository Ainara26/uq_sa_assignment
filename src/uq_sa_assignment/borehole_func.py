import numpy as np

def func(x):
    x = x.T
    r_w, r, t_u, h_u, t_l, h_l, l, k_w = x
    num = 2*np.pi*t_u*(h_u-h_l)
    den = np.log(r/r_w)*(1+((2*l*t_u)/(np.log(r/r_w)*r_w**2*k_w))+(t_u/t_l))

    return num/den
