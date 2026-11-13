import numpy as np

def condense(K, v, T, NMstr, tol=1e-16):
    """
    Keep the first NMstr DOFs and merge all remaining DOFs into one common DOF.

    Parameters
    ----------
    K : (N, N) array_like
    v : (N,) or (N, 1) array_like
    T : (N, M) array_like
    NMstr : int
        Number of master DOFs (kept unchanged).
    tol : float
        Relative tolerance for the size consistency check.

    Returns
    -------
    K_C : (NMstr+1, NMstr+1) ndarray
    v_C : (NMstr+1, 1) ndarray
    T_C : (NMstr+1, M) ndarray
    """
    K = np.asarray(K, dtype=float)
    v = np.asarray(v, dtype=float)
    T = np.asarray(T, dtype=float)

    # Basic shape checks
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square (N x N) matrix.")

    NDoF_k = K.shape[0]
    NDoF_v = v.shape[0]
    NDoF_T = T.shape[0]

    # Size consistency check
    NDoF = (NDoF_k + NDoF_v + NDoF_T) / 3.0
    NDoF_errs = abs(NDoF_k - NDoF) + abs(NDoF_v - NDoF) + abs(NDoF_T - NDoF)
    if NDoF_errs > tol * max(1.0, NDoF):
        raise ValueError(
            "ERROR in condense: Matrix/vector sizes are not consistent. "
            f"The_NDoFs = [K:{NDoF_k}, v:{NDoF_v}, T:{NDoF_T}]"
        )

    NDoF = int(round(NDoF))
    NMstr = int(NMstr)
    Nslav = NDoF - NMstr
    if Nslav < 0:
        raise ValueError("NMstr cannot exceed total number of DOFs (NDoF).")

    # Condensation matrix:
    # [ I(NMstr)      0 ]
    # [ 0        1(Nslav) ]
    T_cndns = np.block([
        [np.eye(NMstr),                np.zeros((NMstr, 1))],
        [np.zeros((Nslav, NMstr)),     np.ones((Nslav, 1))]
    ])

    v_col = v.reshape(-1, 1) if v.ndim == 1 else v

    K_C = T_cndns.T @ (K @ T_cndns)
    v_C = T_cndns.T @ v_col
    T_C = T_cndns.T @ T

    return K_C, v_C, T_C
