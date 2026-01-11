import numpy as np
from numba import njit


################# SEGMENT COST ################
# L2
def L2(seg, T):
    return seg.size * seg.var()

def get_L(T, laplace_option='heat', hyperparam=1): # Graph Laplacian of chain graph
    U = spy.fftpack.dct(np.eye(T), type=2, norm='ortho', axis=0).T # eigenvectors : DCT-II basis functions
    l = 2-2*np.cos(np.pi/T * np.arange(T)) # eigenvalues
    if option == 'dirichlet': # L
        L = np.around(U @ np.diag(l) @ U.T, 3)
        return L
    if option == 'tps': # L^m
        m = hyperparam
        Lm = np.around(U @ np.diag(l)**m @ U.T, 3)
        return Lm
    if option == 'heat': # exp(-tau L)
        tau = hyperparam
        expL = U @ np.diag(np.exp(-tau*l)) @ U.T
        return expL
    

# spectral rbf kernel
@njit
def k(X, Y, sigma=1, type='linear', laplace_option='heat', laplace_hyperparam=1, centered=True, nonlinear_hyperparam=1):
    # X: TxD matrix corresponding to one "chunk" of T times where D features are observed
    # Y: TxD matrix corresponding to one "chunk" of T times where D features are observed
    # sigma: hyperparameter of data kernel k_D: R^D x R^D \to R
    # type: if linear, then k is the linear kernel on the RKHS of the kernel k_T: \{1,\ldots, T\}\times \{1,\ldots,T\}\to R
    #       where k_T is defined by a Gram matrix L^+ (Moore-Penrose pseudoinverse of chain graph Laplacian); (L^+)^m (Moore-Penrose 
    #       pseudoinverse of iterated Laplace matrix); or exp(-tau*L) (heat).
    #       if gaussian or Laplacian, then k is the Gaussian or Laplacian kernel on this RKHS (not to be confused with Laplace matrix)
    # laplace_option: defines base kernel k_T on index set \{1,\ldots, T\}
    # laplace_hyperparam: defines base kernel k_T on index set \{1,\ldots, T\}
    # nonlinear_hyperparam: hyperparameter for nonlinear kernels (Gaussian, Laplacian, etc.)
    # centered: if centered, do HSIC: tr(LHKH) rather than tr(LK). Equivalent for linear kernel on RKHS of L^+ or (L^+)^m but not for linear kernel on RKHS of exp(-tau L)
    # output K(X,Y) is a real number
    
    T = X.shape[0]
    D = X.shape[1]
    
    # "cross Gram matrix" G_{i,j} = k_D(x[i,:], x[j,:])
    x_sq = np.sum(x**2, axis=1, keepdims=True)  # (T, 1)
    y_sq = np.sum(y**2, axis=1, keepdims=True).T # (1, T)
    dist_sq = x_sq + y_sq - 2 * (x @ y.T)        # (T, T) ||x[i,:] - y[j,:]||_{R^d}^2 = ||x[i,:]||_{R^d}^2 + ||y[j,:]||_{R^d}^2 - 2 \langle x[i,:], y[j,:]\rangle
    G = np.exp(-dist_sq/(2*sigma**2)) # G_{i,j} = k_D(x[i,:], y[j,:]) = exp(-||x[i,:] - y[j,:]||_2^2/(2*sigma**2))

    L = get_L(T, laplace_option=laplace_option, laplace_hyperparam=laplace_hyperparam)
    
    if type == 'linear': # kernel is of the form trace(LG) where L is a (variant of) the chain graph Laplacian and G
        if not centered:
            return np.sum(L*G) # avoid matrix multiplication: tr(L@G) = sum_{i,j=1}^T L[i,j]*G[i,j]
        else:
            return np.sum( (L - np.sum(L)/T) * (G - np.sum(G)/T) )

    if type == 'gaussian':
        D = 2*x_sq -2*(x@x.T) + 2*y_sq - 2*(y@y.T) + G
        if not centered:
            return np.exp( np.sum(L*D)/(nonlinear_hyperparam**2) ) # avoid matrix multiplication: tr(L@G) = sum_{i,j=1}^T L[i,j]*G[i,j]
        else:
            return np.exp( np.sum( (L - np.sum(L)/T) * (D - np.sum(D)/T) )/(nonlinear_hyperparam**2) )

    if type == 'laplace':
        D = 2*x_sq -2*(x@x.T) + 2*y_sq - 2*(y@y.T) + G
        if not centered:
            return np.exp(-nonlinear_hyperparam*np.sqrt(np.sum(L*D)) ) # avoid matrix multiplication: tr(L@G) = sum_{i,j=1}^T L[i,j]*G[i,j]
        else:
            return np.exp(-nonlinear_hyperparam*np.sqrt(np.sum( (L - np.sum(L)/T) * (D - np.sum(D)/T) )))

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
