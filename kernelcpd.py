import numpy as np
from numba import njit


################# SEGMENT COST ################
# L2
def L2(seg, T, sigma, laplace_option, laplace_hyperparameter, kernel_type, centered, nonlinear_hyperparam):
    return seg.size * seg.var()


# spectral rbf kernel
@njit
def heat_kernel(L, tau):
    eigvals, eigvecs = np.linalg.eigh(L)
    exp_eigvals = np.exp(-tau * eigvals)
    T = L.shape[0]
    Lm = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            s = 0.0
            for k in range(T):
                s += eigvecs[i, k] * exp_eigvals[k] * eigvecs[j, k]
            Lm[i, j] = s
    return Lm


@njit
def h1_kernel(L, alpha):
    eigvals, eigvecs = np.linalg.eigh(L)
    T = L.shape[0]
    Lm = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            s = 0.0
            for k in range(T):
                s += eigvecs[i, k] * (eigvals[k] + alpha) * eigvecs[j, k]
            Lm[i, j] = s
    return Lm


@njit
def k(X, Y, sigma=1.0, laplace_option='dirichlet', laplace_hyperparameter=1, kernel_type='linear', centered=True, nonlinear_hyperparam=1):
    T, D = X.shape

    # Construct Lm
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
        
    if laplace_option == 'dirichlet':                   # L
        Lm = L
    elif laplace_option == 'tps':                       # L^m
        Lm = L.copy()
        for _ in range(laplace_hyperparameter - 1):
            Lm = Lm @ L
    elif laplace_option == 'heat':                      # exp(-tau L)
        Lm = heat_kernel(L, laplace_hyperparameter)
    elif laplace_option == 'h1':                        # sobolev H1
        Lm = h1_kernel(L, laplace_hyperparameter)

    # Gaussian kernel G
    G = np.zeros((T, T))
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    for i in range(T):
        for j in range(T):
            sq_dist = 0.0
            for d in range(D):
                diff = X[i, d] - Y[j, d]
                sq_dist += diff * diff
            G[i, j] = np.exp(-sq_dist * inv_2sigma2)

    # kernel
    if kernel_type == 'linear':
        result = 0.0
        for i in range(T):
            for j in range(T):
                result += Lm[i, j] * G[j, i]
    else:
        acc = 0.0
        for i in range(T):
            for j in range(T):
                d = 0.0
                for k in range(D):
                    diff = X[i, k] - Y[j, k]
                    d += diff * diff
                acc += Lm[i, j] * d
        if kernel_type == 'gaussian':
            sigma2 = nonlinear_hyperparam * nonlinear_hyperparam
            result = np.exp(-acc / sigma2)
        elif kernel_type == 'laplace':
            gamma = nonlinear_hyperparam
            result = np.exp(-gamma * np.sqrt(acc))
    
    return result


# spectral rbf
@njit
def srbf(seg, T, sigma=1.0, laplace_option='dirichlet', laplace_hyperparameter=1, kernel_type='linear', centered=True, nonlinear_hyperparam=1):
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
        sum_k_xx += k(X_i, X_i, sigma, laplace_option, laplace_hyperparameter, kernel_type, centered, nonlinear_hyperparam)
        for j in range(n):
            X_j = seg_array[j].reshape(-1, 1)
            sum_k_xy += k(X_i, X_j, sigma, laplace_option, laplace_hyperparameter, kernel_type, centered, nonlinear_hyperparam)
    
    var_h = (sum_k_xx / n) - (sum_k_xy / (n * n))
    return n * var_h


################# LIST OF MODEL IN A DICTIONARY ################
Loss = {
    'L2': L2,
    'srbf': srbf,
}


################# DYNAMIC PROGRAMMING ################
def pelt(sequence, pen, model='srbf', T=1, sigma=1.0, laplace_option='dirichlet', laplace_hyperparameter=1, kernel_type='linear', centered=True, nonlinear_hyperparam=1):
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
                value = Loss[model](sequence[tau:t+T-1], T, sigma, laplace_option, laplace_hyperparameter, kernel_type, centered, nonlinear_hyperparam)
            else:
                value = C[tau - T + 1] + pen + Loss[model](sequence[tau:t+T-1], T, sigma, laplace_option, laplace_hyperparameter, kernel_type, centered, nonlinear_hyperparam)
            if value < best_value:
                best_value = value
                best_tau = tau

        C[t] = best_value
        tau_star[t - 1] = best_tau

        # Pruning step for R
        new_R = []
        for tau in R:
            if tau == 0:
                value_no_pen = Loss[model](sequence[tau:t+T-1], T, sigma, laplace_option, laplace_hyperparameter, kernel_type, centered, nonlinear_hyperparam)
            else:
                value_no_pen = C[tau - T + 1] + Loss[model](sequence[tau:t+T-1], T, sigma, laplace_option, laplace_hyperparameter, kernel_type, centered, nonlinear_hyperparam)
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


################# BINARY SEGMENTATION ################

def binseg(sequence, model='srbf', n_changepoints=1, T=1, sigma=1.0, laplace_option='dirichlet', laplace_hyperparameter=1, kernel_type='linear', centered=True, nonlinear_hyperparam=1):
    if model == 'L2':
        T = 1

    n = len(sequence)
    changepoints = []

    def find_best_split(start, end):
        best_cost = np.inf
        best_tau = None

        for tau in range(start + T, end - T + 1):
            cost = (
                Loss[model](sequence[start:tau], T, sigma, laplace_option, laplace_hyperparameter, kernel_type, centered, nonlinear_hyperparam) +
                Loss[model](sequence[tau:end], T, sigma, laplace_option, laplace_hyperparameter, kernel_type, centered, nonlinear_hyperparam)
            )
            if cost < best_cost:
                best_cost = cost
                best_tau = tau

        return best_tau, best_cost

    segments = [(0, n)]

    while len(changepoints) < n_changepoints:
        best_improvement = -np.inf
        best_split = None
        best_segment_idx = None

        for i, (start, end) in enumerate(segments):
            if end - start < 2 * T:
                continue

            current_cost = Loss[model](sequence[start:end], T, sigma, laplace_option, laplace_hyperparameter, kernel_type, centered, nonlinear_hyperparam)
            tau, split_cost = find_best_split(start, end)

            if tau is None:
                continue

            improvement = current_cost - split_cost
            if improvement > best_improvement:
                best_improvement = improvement
                best_split = tau
                best_segment_idx = i

        if best_split is None or best_improvement <= 0:
            break

        start, end = segments.pop(best_segment_idx)
        segments.insert(best_segment_idx, (start, best_split))
        segments.insert(best_segment_idx + 1, (best_split, end))

        changepoints.append(best_split)

    return np.array(sorted(changepoints)) - 1