import numpy as np


# how should the error propagate?
# its all partial derivatives
def calc_error_prop(true_L, true_theta, true_a, dthing, time, wrt="theta_0"):
    if wrt == "theta_0":
        dx_dthing = (
            true_L
            * np.cos(true_theta * np.cos(np.sqrt(true_a / true_L) * time))
            * np.cos(np.sqrt(true_a / true_L) * time)
            * dthing
        )
    if wrt == "L":
        dx_dthing = (
            0.5
            * true_theta
            * time
            * np.sqrt(true_a / true_L)
            * np.sin(time * np.sqrt(true_a / true_L))
            * np.cos(true_theta * np.cos(time * np.sqrt(true_a / true_L)))
            + np.sin(true_theta * np.cos(time * np.sqrt(true_a / true_L)))
        ) * dthing
    if wrt == "a_g":
        dx_dthing = (
            -0.5
            * np.sqrt(true_L / true_a)
            * true_theta
            * time
            * np.sin(np.sqrt(true_a / true_L) * time)
            * np.cos(true_theta * np.cos(np.sqrt(true_a / true_L) * time))
        ) * dthing
    if wrt == "all":
        dx_dthing = (
            true_L
            * np.cos(true_theta * np.cos(np.sqrt(true_a / true_L) * time))
            * np.cos(np.sqrt(true_a / true_L) * time)
            * dthing[1]
            + (
                0.5
                * true_theta
                * time
                * np.sqrt(true_a / true_L)
                * np.sin(time * np.sqrt(true_a / true_L))
                * np.cos(true_theta * np.cos(time * np.sqrt(true_a / true_L)))
                + np.sin(true_theta * np.cos(time * np.sqrt(true_a / true_L)))
            )
            * dthing[0]
            + (
                -0.5
                * np.sqrt(true_L / true_a)
                * true_theta
                * time
                * np.sin(np.sqrt(true_a / true_L) * time)
                * np.cos(true_theta * np.cos(np.sqrt(true_a / true_L) * time))
            )
            * dthing[2]
        )
    return abs(dx_dthing)
