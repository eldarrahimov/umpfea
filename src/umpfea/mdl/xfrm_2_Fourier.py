"""
Nodal to Fourier basis transform for a stator or rotor compact model.

Output DOF ordering (size 2*Nh+2):
  0           : DC
  1 .. Nh     : cos(ih*theta)
  Nh+1 .. 2Nh : sin(ih*theta)
  2*Nh+1      : common node, passed through unchanged
"""

import numpy as np


def xfrm_2_Fourier(Nods_XY, Nh, K_mat, v_PM, T_mat):
    """
    Transform nodal K, v_PM, T to the Fourier basis.

    Parameters
    ----------
    Nods_XY : (NNod, 2) ndarray
        Coordinates of the airgap-facing master nodes (no row for the
        common node).
    Nh : int
        Number of harmonics kept (1..Nh).
    K_mat : (NNod+1, NNod+1) ndarray
        Nodal stiffness; last DOF is the common node.
    v_PM : (NNod+1,) or (NNod+1, 1) ndarray
    T_mat : (NNod+1, NCC) ndarray

    Returns
    -------
    K_mat_F : (2*Nh+2, 2*Nh+2) ndarray
    v_PM_F  : (2*Nh+2, 1) ndarray
    T_mat_F : (2*Nh+2, NCC) ndarray
    """
    NNod = Nods_XY.shape[0]
    Nd_angls = np.arctan2(Nods_XY[:, 1], Nods_XY[:, 0])   # (NNod,) angles

    NF = 2 * Nh + 2
    Tt = np.zeros((NNod + 1, NF))

    # DC column
    Tt[:NNod, 0] = 1.0

    # Cosine and sine columns
    for ih in range(1, Nh + 1):
        Tt[:NNod, ih]      = np.cos(ih * Nd_angls)
        Tt[:NNod, Nh + ih] = np.sin(ih * Nd_angls)

    # Common node maps to the last Fourier DOF
    Tt[NNod, NF - 1] = 1.0

    v_col = np.asarray(v_PM, dtype=float)
    if v_col.ndim == 1:
        v_col = v_col[:, None]

    K_mat_F = Tt.T @ K_mat @ Tt          # (NF, NF)
    v_PM_F  = Tt.T @ v_col               # (NF, 1)
    T_mat_F = Tt.T @ T_mat               # (NF, NCC)

    return K_mat_F, v_PM_F, T_mat_F
