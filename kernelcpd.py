import numpy as np
from numba import njit

@njit
def k(X, Y, sigma=10.0):
    T = X.shape[0]
    D = X.shape[1]

    # generate L_chain
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

    # generate G
    G = np.zeros((T, T))
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    for i in range(T):
        for j in range(T):
            sq_dist = 0.0
            for d in range(D):
                diff = X[i, d] - Y[j, d]
                sq_dist += diff * diff
            G[i, j] = np.exp(-sq_dist * inv_2sigma2)

    # trace(L @ G)
    result = 0.0
    for i in range(T):
        for j in range(T):
            result += L[i, j] * G[j, i]

    return result


@njit
def pairwise_sum(Y, indices):
    s = 0.0
    m = indices.shape[0]

    for a in range(m):
        i = indices[a]
        for b in range(a + 1, m):
            j = indices[b]
            s += k(Y[i], Y[j])

    return 2.0 * s

@njit
def cross_sum(Y, L, R):
    cross = 0.0
    for a in range(L.shape[0]):
        i = L[a]
        Yi = Y[i]
        for b in range(R.shape[0]):
            j = R[b]
            cross += k(Yi, Y[j])
    return cross


# algorithm
def spectral_cpd(y: np.ndarray, T: int) -> int:
    """
    Parameters
    ----------
    y : np.ndarray
        observation of shape (n, d)
    T : int
        chunk size

    Returns
    -------
    int
        split point location
    """

    # ---- inputs validation ----
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")

    if y.ndim != 2:
        raise ValueError("y must have shape (n, d)")

    n, d = y.shape
    if n == 0 or d == 0:
        raise ValueError("y must be non-empty")

    if not isinstance(T, int):
        raise TypeError("T must be an integer")

    if T <= 0:
        raise ValueError("T must be positive")

    if not callable(k):
        raise TypeError("k must be a kernel")

    # ---- algorithm ----
    Y = [y[i:i+T] for i in range(len(y))]

    maxMMD = -np.inf
    s_star = None

    for s in range(T, n - T - 1):
        L = np.arange(s - T + 2)
        R = np.arange(s + 1, n - T + 1)
        mL, mR = len(L), len(R)

        term_L = pairwise_sum(Y, L) / (mL * (mL - 1))
        term_R = pairwise_sum(Y, R) / (mR * (mR - 1))
        term_LR = cross_sum(Y, L, R) / (mL * mR)

        MMD = term_L + term_R - 2.0 * term_LR
        if MMD > maxMMD:
            maxMMD = MMD
            s_star = s
    
    return s_star