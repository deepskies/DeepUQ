# how should this error propagate?

def calc_error_prop(true_L, true_theta, true_a, dthing, time = 0.5, wrt = 'theta_0'):
    if wrt == 'theta_0':
        dx_dthing = true_L * np.cos(true_theta * np.cos(np.sqrt(true_a / true_L) * one_time)) * \
              np.cos(np.sqrt(true_a / true_L) * one_time) * dthing
    if wrt == 'L':

        dx_dthing = (0.5 * true_theta * time * np.sqrt(true_a / true_L) * np.sin(time * np.sqrt(true_a / true_L)) * \
             np.cos(true_theta * np.cos(time * np.sqrt(true_a / true_L))) + \
             np.sin(true_theta * np.cos(time * np.sqrt(true_a / true_L)))) * dthing
    return dx_dthing
