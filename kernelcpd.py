import numpy as np

# kernel
def k(X, Y, sigma=1.0):
    sq_norms = np.sum((X - Y) ** 2, axis=1)
    k = np.exp(-sq_norms / (2 * sigma ** 2))
    return np.mean(k)


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
    maxMMD = 0
    s_star = 0
    for s in range(T - 1, n - T):
        L, R = np.arange(s), np.arange(s, n - T)
        mL, mR = s + 1, n - s
        MMD = sum(k(y[i:i+T], y[j:j+T]) for i in L for j in L if i != j) / (mL * (mL - 1)) \
            + sum(k(y[i:i+T], y[j:j+T]) for i in R for j in R if i != j) / (mR * (mR - 1)) \
            - 2 * sum(k(y[i:i+T], y[j:j+T]) for i in L for j in R) / (mL * mR)
        if MMD > maxMMD:
            maxMMD = MMD
            s_star = s
    return s_star