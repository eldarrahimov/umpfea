"""
Forces, back-EMFs and inductances from CompactModel1 (Fourier basis).

The airgap stiffness comes from the precomputed tables (calc_agapi /
calc_agapj), so no airgap remeshing is needed per call.

DOF ordering per body (size 2*Nh+2): DC, cos 1..Nh, sin 1..Nh, common node.
Rotor DOFs come first, then stator DOFs.
"""

import numpy as np

from umpfea.mdl.agap import calc_agapi, calc_agapj


def analyse1(compact_model, rotr_state, cct_currs):
    """
    Forces, back-EMFs and inductances at one rotor state.

    Parameters
    ----------
    compact_model : CompactModel1
    rotr_state : (6,) array_like
        [x (m), y (m), theta (rad), dx/dt, dy/dt, dtheta/dt]
    cct_currs : (NCC,) array_like
        Circuit currents (A).

    Returns
    -------
    Frcs      : (3,) ndarray   [Fx (N), Fy (N), torque (N m)]
    VCC       : (NCC,) ndarray motional back-EMF (V)
    LCC       : (NCC, NCC) ndarray inductance matrix (H)
    Avec0     : (NDoF,) ndarray potentials (Fourier coefficients)
    d_Avec_dt : (NDoF,) ndarray rate of change of potentials
    """
    K_matSS_F = compact_model.K_matSS_F
    v_PM_S_F  = np.asarray(compact_model.v_PM_S_F, dtype=float).reshape(-1, 1)
    T_mat_S_F = compact_model.T_mat_S_F
    Nh_S      = compact_model.Nh_S
    K_matRR_F = compact_model.K_matRR_F
    v_PM_R_F  = np.asarray(compact_model.v_PM_R_F, dtype=float).reshape(-1, 1)
    T_mat_R_F = compact_model.T_mat_R_F
    Nh_R      = compact_model.Nh_R
    Eccs      = compact_model.Eccs
    Lmac      = compact_model.Lmac

    NDoF_R = K_matRR_F.shape[0]   # 2*Nh_R + 2
    NDoF_S = K_matSS_F.shape[0]   # 2*Nh_S + 2
    NDoF   = NDoF_R + NDoF_S
    NCC_R  = T_mat_R_F.shape[1]
    NCC_S  = T_mat_S_F.shape[1]
    assert NCC_R == NCC_S, f"NCC mismatch: rotor {NCC_R} vs stator {NCC_S}"
    NCC = NCC_R

    u_rotr = float(rotr_state[0])
    v_rotr = float(rotr_state[1])
    q_rotr = float(rotr_state[2])
    ecc    = float(np.sqrt(u_rotr**2 + v_rotr**2))
    q_ecc  = float(np.arctan2(v_rotr, u_rotr)) if ecc > 0.0 else 0.0

    # Rotor and stator blocks
    K_mat_sys0 = np.zeros((NDoF, NDoF))
    K_mat_sys0[:NDoF_R, :NDoF_R] = K_matRR_F
    K_mat_sys0[NDoF_R:, NDoF_R:] = K_matSS_F

    # Airgap DOFs: everything except the two common nodes
    irng_R = np.arange(0,      2*Nh_R + 1)
    irng_S = np.arange(NDoF_R, NDoF_R + 2*Nh_S + 1)
    irange = np.concatenate([irng_R, irng_S])
    kt     = NDoF - 1                       # stator common node

    # Gauge: stiffen the stator common node (as in analyse0)
    K_mat_sys0[kt, kt] *= 2.0

    Kc0, Ks0, d_Kc_d_e, d_Ks_d_e = calc_agapi(
        Eccs, compact_model.KcosN, compact_model.KsinN, Nh_R, Nh_S, ecc
    )

    K_AG, d_Kag_d_ecc, d_Kag_d_qe, d_Kag_d_qr, d_Kag_d_n = calc_agapj(
        Kc0, Ks0, d_Kc_d_e, d_Ks_d_e, Nh_R, Nh_S, q_ecc, q_rotr
    )

    K_mat_sys0[np.ix_(irange, irange)] += K_AG

    v_PM_sys  = np.vstack([v_PM_R_F, v_PM_S_F])       # (NDoF, 1)
    T_mat_sys = np.vstack([T_mat_R_F, T_mat_S_F])     # (NDoF, NCC)
    v_exctn   = v_PM_sys + T_mat_sys @ np.asarray(cct_currs, dtype=float).reshape(-1, 1)  # (NDoF, 1)

    Avec0 = np.linalg.solve(K_mat_sys0, v_exctn)       # (NDoF, 1)

    # Airgap derivatives embedded in full-size matrices
    d_Kmat_d_ecc = np.zeros((NDoF, NDoF))
    d_Kmat_d_ecc[np.ix_(irange, irange)] = d_Kag_d_ecc
    d_Kmat_d_qe  = np.zeros((NDoF, NDoF))
    d_Kmat_d_qe[np.ix_(irange, irange)]  = d_Kag_d_qe
    d_Kmat_d_qr  = np.zeros((NDoF, NDoF))
    d_Kmat_d_qr[np.ix_(irange, irange)]  = d_Kag_d_qr
    d_Kmat_d_n   = np.zeros((NDoF, NDoF))
    d_Kmat_d_n[np.ix_(irange, irange)]   = d_Kag_d_n

    # One multi-RHS solve: columns 0..2 give dA/d(position) via
    # K dA = -dK A0, the remaining NCC columns give K^-1 T for LCC.
    if ecc > 0.0:
        TX_t = np.hstack([d_Kmat_d_ecc @ Avec0,
                          d_Kmat_d_qe  @ Avec0,
                          d_Kmat_d_qr  @ Avec0])
        TX = np.hstack([TX_t, T_mat_sys])
        TY = np.linalg.solve(K_mat_sys0, TX)

        d_Avec_d_ecc = -TY[:, 0:1]
        d_Avec_d_qe  = -TY[:, 1:2]
        d_Avec_d_qr  = -TY[:, 2:3]

        # Polar (ecc, q_ecc) to Cartesian (x, y)
        d_Avec_d_x = (  d_Avec_d_ecc * np.cos(q_ecc)
                      - d_Avec_d_qe  * (np.sin(q_ecc) / ecc))
        d_Avec_d_y = (  d_Avec_d_ecc * np.sin(q_ecc)
                      + d_Avec_d_qe  * (np.cos(q_ecc) / ecc))
    else:
        # At ecc = 0 (q_ecc = 0): d/d_ecc is d/dx and d/dn is d/dy
        TX_t = np.hstack([d_Kmat_d_ecc @ Avec0,
                          d_Kmat_d_n   @ Avec0,
                          d_Kmat_d_qr  @ Avec0])
        TX = np.hstack([TX_t, T_mat_sys])
        TY = np.linalg.solve(K_mat_sys0, TX)

        d_Avec_d_x  = -TY[:, 0:1]
        d_Avec_d_y  = -TY[:, 1:2]
        d_Avec_d_qr = -TY[:, 2:3]

    vel = np.asarray(rotr_state[3:6], dtype=float)
    d_Avec_dt = (  d_Avec_d_x   * vel[0]
                 + d_Avec_d_y   * vel[1]
                 + d_Avec_d_qr  * vel[2])          # (NDoF, 1)

    VCC = (T_mat_sys.T @ d_Avec_dt).flatten() * Lmac   # (NCC,)

    LCC = T_mat_sys.T @ TY[:, 3:]                  # (NCC, NCC)
    LCC = (LCC + LCC.T) / 2.0 * Lmac

    # Virtual work: F_i = v_exctn' * dA/dq_i * Lmac/2
    ve = v_exctn.ravel()
    Frcs = np.array([
        ve @ d_Avec_d_x.ravel(),
        ve @ d_Avec_d_y.ravel(),
        ve @ d_Avec_d_qr.ravel(),
    ]) * (Lmac / 2.0)

    return Frcs, VCC, LCC, Avec0.flatten(), d_Avec_dt.flatten()
