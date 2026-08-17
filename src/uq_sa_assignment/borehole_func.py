import numpy as np

def func(x, args):
    t_u, h_u, h_l, r, r_w, l, t_l, k_w = args

    A = 2*np.pi*t_u*(h_u-h_l)
    B = np.log(r/r_w)*(1+(2*l*t_u/np.log(r/r_w)*r_w**2*k_w)+(t_u/t_l))

    return A/B
