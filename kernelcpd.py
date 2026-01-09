import numpy as np
from numba import njit


################# SEGMENT COST ################
# L2
def L2(seg, T):
    return seg.size * seg.var()

# spectral rbf kernel
@njit
def k(X, Y, sigma=1.0, m=1.0):
    T = X.shape[0]
    D = X.shape[1]

    L = np.zeros((T, T))
    for i in range(T):
        if i == 0 or i == T - 1:
            L[i, i] = 1.0
        else:
            L[i, i] = 2.0
        if i - 1 >= 0:
            L[i, i - 1] = -1.0
        if i + 1 < T:
            L[i, i + 1] = -1.0

    G = np.zeros((T, T))
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    for i in range(T):
        for j in range(T):
            sq_dist = 0.0
            for d in range(D):
                diff = X[i, d] - Y[j, d]
                sq_dist += diff * diff
            G[i, j] = np.exp(-sq_dist * inv_2sigma2)

    result = 0.0
    for i in range(T):
        for j in range(T):
            result += L[i, j]**m * G[j, i]

    return result

# spectral rbf
@njit
def srbf(seg, T):
    n_total, d = seg.shape
    n_seg = n_total - T + 1
    seg_array = np.empty((n_seg, T, d))

    # Build rolling segments
    for i in range(n_seg):
        seg_array[i] = seg[i:i+T]

    n = seg_array.shape[0]

    sum_k_xx = 0.0
    sum_k_xy = 0.0

    for i in range(n):
        X_i = seg_array[i].reshape(-1, 1)
        sum_k_xx += k(X_i, X_i)
        for j in range(n):
            X_j = seg_array[j].reshape(-1, 1)
            sum_k_xy += k(X_i, X_j)
    
    var_h = (sum_k_xx / n) - (sum_k_xy / (n * n))
    return n * var_h


################# LIST OF MODEL IN A DICTIONARY ################
L = {
    'L2': L2,
    'srbf': srbf,
}


################# DYNAMIC PROGRAMMING ################
def pelt(sequence, pen, model, T=1):
    if model == 'L2':
        T = 1
    
    n = len(sequence)

    # Cost array
    C = np.zeros(n + 1)

    # Backpointer
    tau_star = np.zeros(n, dtype=int)

    # Set of andidate changepoints R
    R = [0]

    # Dymanic programming
    for t in range(T, n + 2 - T):
        best_value = np.inf
        best_tau = 0
        
        # Evaluate all candidates and choose the best one
        for tau in R:
            if tau == 0:
                value = L[model](sequence[tau:t+T-1], T)
            else:
                value = C[tau - T + 1] + pen + L[model](sequence[tau:t+T-1], T)
            if value < best_value:
                best_value = value
                best_tau = tau

        C[t] = best_value
        tau_star[t - 1] = best_tau

        # Pruning step for R
        new_R = []
        for tau in R:
            if tau == 0:
                value_no_pen = L[model](sequence[tau:t+T-1], T)
            else:
                value_no_pen = C[tau - T + 1] + L[model](sequence[tau:t+T-1], T)
            if value_no_pen <= C[t]:
                new_R.append(tau)

        new_R.append(t)
        R = new_R

    tau_star[n-T+1:n] = tau_star[n-T]

    set_of_chpnt = trace_back(tau_star)
    return set_of_chpnt - 1


def trace_back(tau_star):
    tau = tau_star[-1]
    chpnt = np.array([len(tau_star)], dtype=int)
    while tau > 0:
        chpnt = np.append(tau, chpnt)
        tau = tau_star[tau-1]
    return chpnt[:-1]