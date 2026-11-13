"""
Forces, back-EMFs and inductances from CompactModel0 (nodal basis), with
the airgap meshed by stitch_agap at each call.

Node labels in stitch_agap / agap_build:
  label < NLOff            stator node, index = label
  NLOff <= label < 2*NLOff rotor node,  index = label - NLOff
  label >= 2*NLOff         mid-gap node, index = label - 2*NLOff

Column 2 of MGNdsXY holds 0-based rotor node indices.
Lengths in metres.
"""

import numpy as np

from umpfea.mdl.build_k import ele6AG_calx


def move_rotor(MNodes_R, u_rotr, v_rotr, q_rotr):
    """
    Rotate rotor nodes by q_rotr (rad, anticlockwise) about the origin, then
    translate by (u_rotr, v_rotr) (m).

    Returns
    -------
    MNodes_RT : (NR, 2) ndarray
    """
    c, s  = np.cos(q_rotr), np.sin(q_rotr)
    Rotn  = np.array([[c, -s], [s, c]])
    MNodes_RT = MNodes_R @ Rotn.T
    MNodes_RT = MNodes_RT.copy()
    MNodes_RT[:, 0] += u_rotr
    MNodes_RT[:, 1] += v_rotr
    return MNodes_RT


def stitch_agap(Stator_XY, Rotor_XY, NLOff):
    """
    Mesh the airgap with (NS+NR)//2 six-node triangles.

    Both node lists must be ordered anticlockwise, alternating corner (even
    index) and mid-side (odd index). Each step adds a triangle on the stator
    or rotor side, whichever gives the shorter diagonal.

    Parameters
    ----------
    Stator_XY : (NS, 2) ndarray, stator bore nodes
    Rotor_XY  : (NR, 2) ndarray, rotor OD nodes
    NLOff     : int, label offset, >= max(NS, NR)

    Returns
    -------
    Elems3   : ((NS+NR)//2, 6) int ndarray, node labels
    MGNdsXY  : ((NS+NR)//2, 3) ndarray, mid-gap nodes [x, y, rotor node index];
               each mid-gap node follows that rotor node in budge_ag_nds
    sttr_rad : float, mean stator bore radius (m)
    clrnce   : float, minimum clearance (m)
    """
    NS    = Stator_XY.shape[0]
    NR    = Rotor_XY.shape[0]
    NEle3 = (NS + NR) // 2

    if NLOff < max(NS, NR):
        raise ValueError(
            f'stitch_agap: NLOff={NLOff} must be > max(NS={NS}, NR={NR})')

    Elems3  = np.zeros((NEle3, 6), dtype=int)
    MGNdsXY = np.zeros((NEle3, 3))

    sttr_rad    = np.mean(np.sqrt(Stator_XY[:, 0]**2 + Stator_XY[:, 1]**2))
    rotr_rad_mx = np.max( np.sqrt(Rotor_XY[:,  0]**2 + Rotor_XY[:,  1]**2))
    clrnce      = sttr_rad - rotr_rad_mx
    if clrnce < sttr_rad * 1e-6:
        raise ValueError(
            f'stitch_agap: rotor penetrating stator '
            f'(sttr_rad={sttr_rad*1e3:.4f} mm, '
            f'rotr_rad_mx={rotr_rad_mx*1e3:.4f} mm)')

    # Start at stator node 0 and the nearest rotor corner
    i_sttr_nd   = 0
    sttr_ndi_xy = Stator_XY[i_sttr_nd, :]

    Rotor_crnr_XY = Rotor_XY[0::2, :]   # 0, 2, 4, ... are corner nodes
    dists         = np.sqrt((Rotor_crnr_XY[:, 0] - sttr_ndi_xy[0])**2
                            + (Rotor_crnr_XY[:, 1] - sttr_ndi_xy[1])**2)
    i_rotr_nd   = 2 * int(np.argmin(dists))   # index in the full rotor array
    rotr_ndi_xy = Rotor_XY[i_rotr_nd, :]

    i_midg_nd = 0
    MGNdsXY[i_midg_nd, :] = [
        (sttr_ndi_xy[0] + rotr_ndi_xy[0]) / 2,
        (sttr_ndi_xy[1] + rotr_ndi_xy[1]) / 2,
        float(i_rotr_nd)]

    # One element per iteration
    for iele in range(NEle3):
        j_sttr_nd   = (i_sttr_nd + 2) % NS      # next stator corner
        sttr_ndj_xy = Stator_XY[j_sttr_nd, :]
        j_rotr_nd   = (i_rotr_nd + 2) % NR      # next rotor corner
        rotr_ndj_xy = Rotor_XY[j_rotr_nd, :]

        scor1 = np.linalg.norm(rotr_ndj_xy - sttr_ndi_xy)
        scor2 = np.linalg.norm(rotr_ndi_xy - sttr_ndj_xy)

        j_midg_nd = (i_midg_nd + 1) % NEle3

        if scor1 > scor2:
            # Base on the stator side
            midnode_xy = (rotr_ndi_xy + sttr_ndj_xy) / 2
            Elems3[iele, :] = [
                i_sttr_nd,                  # stator corner
                i_rotr_nd + NLOff,          # rotor corner
                j_sttr_nd,                  # stator corner
                i_midg_nd + 2 * NLOff,      # mid-gap
                j_midg_nd + 2 * NLOff,      # mid-gap
                i_sttr_nd + 1]              # stator midside
            midnode    = np.array([midnode_xy[0], midnode_xy[1], float(i_rotr_nd)])
            i_sttr_nd   = j_sttr_nd
            sttr_ndi_xy = sttr_ndj_xy
        else:
            # Base on the rotor side
            midnode_xy = (sttr_ndi_xy + rotr_ndj_xy) / 2
            Elems3[iele, :] = [
                j_rotr_nd + NLOff,          # rotor corner
                i_sttr_nd,                  # stator corner
                i_rotr_nd + NLOff,          # rotor corner
                j_midg_nd + 2 * NLOff,      # mid-gap
                i_midg_nd + 2 * NLOff,      # mid-gap
                i_rotr_nd + 1 + NLOff]      # rotor midside
            midnode    = np.array([midnode_xy[0], midnode_xy[1], float(j_rotr_nd)])
            i_rotr_nd   = j_rotr_nd
            rotr_ndi_xy = rotr_ndj_xy

        i_midg_nd += 1
        if i_midg_nd < NEle3:
            MGNdsXY[i_midg_nd, :] = midnode

    return Elems3, MGNdsXY, sttr_rad, clrnce


def budge_ag_nds(MGNdsXY, MNodes_R_del):
    """
    Move each mid-gap node by half the displacement of its rotor node.

    Parameters
    ----------
    MGNdsXY      : (NAG, 3) ndarray, [x, y, rotor node index]
    MNodes_R_del : (NR, 2) ndarray, rotor node displacements (m)

    Returns
    -------
    MGNdsXYn : (NAG, 3) ndarray
    """
    MGNdsXYn = MGNdsXY.copy()
    for inod in range(MGNdsXY.shape[0]):
        jnod = int(MGNdsXY[inod, 2])
        MGNdsXYn[inod, :2] += 0.5 * MNodes_R_del[jnod, :]
    return MGNdsXYn


def agap_build(MNodes_S, MNodes_R, MGNdsXY, Elems3, NLOff):
    """
    Assemble the airgap stiffness from the stitch_agap mesh.

    DOF order: rotor nodes + rotor common (NNR+1), mid-gap nodes (NAG),
    stator nodes + stator common (NNS+1). The common DOFs get no airgap
    terms; they are there so the matrix lines up with the compact model.

    Parameters
    ----------
    MNodes_S : (NNS, 2) ndarray, stator bore nodes (m)
    MNodes_R : (NNR, 2) ndarray, rotor OD nodes (m)
    MGNdsXY  : (NAG, 3) ndarray, mid-gap nodes
    Elems3   : (NEle, 6) int ndarray, labels from stitch_agap
    NLOff    : int, label offset used in stitch_agap

    Returns
    -------
    K_mat_AG : (NDoF, NDoF) ndarray, NDoF = (NNR+1) + NAG + (NNS+1)
    """
    NNS  = MNodes_S.shape[0]
    NNR  = MNodes_R.shape[0]
    NAG  = MGNdsXY.shape[0]
    NDoF = (NNR + 1) + NAG + (NNS + 1)

    K_mat_AG = np.zeros((NDoF, NDoF))

    OffstR = 0
    OffstM = NNR + 1
    OffstS = NNR + 1 + NAG

    pts    = np.zeros((6, 2))
    dofmap = np.zeros(6, dtype=int)

    for iele in range(Elems3.shape[0]):
        indxs = Elems3[iele, :]
        for jt in range(6):
            it = int(indxs[jt])
            if it < NLOff:               # stator node
                pts[jt, :]  = MNodes_S[it, :]
                dofmap[jt]  = it + OffstS
            elif it < 2 * NLOff:         # rotor node
                kt = it - NLOff
                pts[jt, :]  = MNodes_R[kt, :]
                dofmap[jt]  = kt + OffstR
            else:                        # mid-gap node
                kt = it - 2 * NLOff
                pts[jt, :]  = MGNdsXY[kt, :2]
                dofmap[jt]  = kt + OffstM

        k_e = ele6AG_calx(pts[:, 0], pts[:, 1])
        K_mat_AG[np.ix_(dofmap, dofmap)] += k_e

    return K_mat_AG


def analyse0(compact_model, rotr_state, cct_currs):
    """
    Forces, back-EMFs and inductances at one rotor state (nodal model).

    Position derivatives are central differences: the airgap is rebuilt at
    +/- a small step in x, y and theta.

    Parameters
    ----------
    compact_model : CompactModel0
    rotr_state : (6,) array_like
        [x (m), y (m), theta (rad), dx/dt, dy/dt, dtheta/dt]
    cct_currs : (NCC,) array_like
        Circuit currents (A).

    Returns
    -------
    Frcs      : (3,) ndarray   [Fx (N), Fy (N), torque (N m)]
    VCC       : (NCC,) ndarray motional back-EMF (V)
    LCC       : (NCC, NCC) ndarray inductance matrix (H)
    Avec0     : (NDoF,) ndarray nodal potentials
    d_Avec_dt : (NDoF,) ndarray rate of change of potentials

    NDoF = (NNR+1) + NAG + (NNS+1), NAG = (NNR+NNS)//2. Forces are positive
    in +x, +y, +theta: F = -A' (dK/dq) A * Lmac/2.
    """
    MNodes_S = compact_model.MNodes_S
    MNodes_R = compact_model.MNodes_R
    Lmac     = compact_model.Lmac
    NNS      = MNodes_S.shape[0]
    NNR      = MNodes_R.shape[0]

    NCC_R = compact_model.T_mat_R.shape[1]
    NCC_S = compact_model.T_mat_S.shape[1]
    if NCC_R != NCC_S:
        raise ValueError(
            f'analyse0: NCC mismatch — rotor has {NCC_R}, stator has {NCC_S}')
    NCC = NCC_R

    NAG   = (NNR + NNS) // 2
    NDoF  = (NNR + 1) + NAG + (NNS + 1)

    # Label offset larger than any node count
    NLOff = 1000 * (((NNS + NNR + 2) + 999) // 1000 + 1)

    rotr_state = np.asarray(rotr_state, dtype=float).flatten()
    cct_currs  = np.asarray(cct_currs,  dtype=float).flatten()

    u_rotr = rotr_state[0]
    v_rotr = rotr_state[1]
    q_rotr = rotr_state[2]

    # Airgap mesh at the actual rotor position
    MNodes_RT0 = move_rotor(MNodes_R, u_rotr, v_rotr, q_rotr)
    Elems3, MNodes_GT0, sttr_rad, clrnce = stitch_agap(
        MNodes_S, MNodes_RT0, NLOff)

    K_mat_sys0 = agap_build(MNodes_S, MNodes_RT0, MNodes_GT0, Elems3, NLOff)

    # Finite-difference steps
    del_t = clrnce / 100          # translation (m)
    del_q = del_t / sttr_rad      # rotation (rad)

    # Perturbed positions; same airgap topology, only the nodes move
    MNodes_RT1 = move_rotor(MNodes_RT0, -del_t,  0.0,    0.0   )
    MNodes_RT2 = move_rotor(MNodes_RT0, +del_t,  0.0,    0.0   )
    MNodes_RT3 = move_rotor(MNodes_RT0,  0.0,   -del_t,  0.0   )
    MNodes_RT4 = move_rotor(MNodes_RT0,  0.0,   +del_t,  0.0   )
    MNodes_RT5 = move_rotor(MNodes_RT0,  0.0,    0.0,   -del_q )
    MNodes_RT6 = move_rotor(MNodes_RT0,  0.0,    0.0,   +del_q )

    MNodes_GT1 = budge_ag_nds(MNodes_GT0, MNodes_RT1 - MNodes_RT0)
    MNodes_GT2 = budge_ag_nds(MNodes_GT0, MNodes_RT2 - MNodes_RT0)
    MNodes_GT3 = budge_ag_nds(MNodes_GT0, MNodes_RT3 - MNodes_RT0)
    MNodes_GT4 = budge_ag_nds(MNodes_GT0, MNodes_RT4 - MNodes_RT0)
    MNodes_GT5 = budge_ag_nds(MNodes_GT0, MNodes_RT5 - MNodes_RT0)
    MNodes_GT6 = budge_ag_nds(MNodes_GT0, MNodes_RT6 - MNodes_RT0)
    
    K_mat_sys1 = agap_build(MNodes_S, MNodes_RT1, MNodes_GT1, Elems3, NLOff)
    K_mat_sys2 = agap_build(MNodes_S, MNodes_RT2, MNodes_GT2, Elems3, NLOff)
    K_mat_sys3 = agap_build(MNodes_S, MNodes_RT3, MNodes_GT3, Elems3, NLOff)
    K_mat_sys4 = agap_build(MNodes_S, MNodes_RT4, MNodes_GT4, Elems3, NLOff)
    K_mat_sys5 = agap_build(MNodes_S, MNodes_RT5, MNodes_GT5, Elems3, NLOff)
    K_mat_sys6 = agap_build(MNodes_S, MNodes_RT6, MNodes_GT6, Elems3, NLOff)
    # Add the rotor and stator compact matrices
    irng_R = np.arange(0, NNR + 1)               # rotor DOFs (incl. common)
    irng_S = np.arange(NNR + 1 + NAG, NDoF)      # stator DOFs (incl. common)
    kt     = NDoF - 1                             # stator common DOF
    tx     = float(compact_model.K_matSS[-1, -1])

    for K_sys in (K_mat_sys0, K_mat_sys1, K_mat_sys2,
                  K_mat_sys3, K_mat_sys4, K_mat_sys5, K_mat_sys6):
        K_sys[np.ix_(irng_R, irng_R)] += compact_model.K_matRR
        K_sys[np.ix_(irng_S, irng_S)] += compact_model.K_matSS
        K_sys[kt, kt]                  += tx    # gauge: stiffen the stator common node

    v_PM_R    = np.asarray(compact_model.v_PM_R, dtype=float).reshape(-1, 1)
    v_PM_S    = np.asarray(compact_model.v_PM_S, dtype=float).reshape(-1, 1)
    v_PM_sys  = np.vstack([v_PM_R, np.zeros((NAG, 1)), v_PM_S])   # (NDoF, 1)
    T_mat_sys = np.vstack([compact_model.T_mat_R,
                            np.zeros((NAG, NCC)),
                            compact_model.T_mat_S])                # (NDoF, NCC)

    d_K_du = (K_mat_sys2 - K_mat_sys1) / (2.0 * del_t)
    d_K_dv = (K_mat_sys4 - K_mat_sys3) / (2.0 * del_t)
    d_K_dq = (K_mat_sys6 - K_mat_sys5) / (2.0 * del_q)

    rhs = v_PM_sys + T_mat_sys @ cct_currs.reshape(-1, 1)   # (NDoF, 1)

    def _solve(K):
        return np.linalg.solve(K, rhs)

    Avec0 = _solve(K_mat_sys0)
    Avec1 = _solve(K_mat_sys1)
    Avec2 = _solve(K_mat_sys2)
    Avec3 = _solve(K_mat_sys3)
    Avec4 = _solve(K_mat_sys4)
    Avec5 = _solve(K_mat_sys5)
    Avec6 = _solve(K_mat_sys6)

    d_Avec_du = (Avec2 - Avec1) / (2.0 * del_t)
    d_Avec_dv = (Avec4 - Avec3) / (2.0 * del_t)
    d_Avec_dq = (Avec6 - Avec5) / (2.0 * del_q)

    d_Avec_dt = (d_Avec_du * rotr_state[3]
                 + d_Avec_dv * rotr_state[4]
                 + d_Avec_dq * rotr_state[5])

    VCC = (T_mat_sys.T @ d_Avec_dt).flatten() * Lmac

    LCC = T_mat_sys.T @ np.linalg.solve(K_mat_sys0, T_mat_sys)
    LCC = (LCC + LCC.T) / 2.0 * Lmac

    # Virtual work
    Yc1  = -(d_K_du @ Avec0)
    Yc2  = -(d_K_dv @ Avec0)
    Yc3  = -(d_K_dq @ Avec0)
    Frcs = np.array([(Yc1.ravel() @ Avec0.ravel()),
                     (Yc2.ravel() @ Avec0.ravel()),
                     (Yc3.ravel() @ Avec0.ravel())]) * (Lmac / 2.0)

    return Frcs, VCC, LCC, Avec0.flatten(), d_Avec_dt.flatten()
