"""
Airgap stiffness in the Fourier basis (used by macbuild1 / analyse1).

  calc_agap1  Kcos, Ksin at one eccentricity
  calc_agapN  tables of Kcos, Ksin over N eccentricities
  calc_agapi  interpolation in eccentricity
  calc_agapj  rotation to the current rotor / eccentricity angles, with
              derivatives

Lengths in metres.
"""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from umpfea.geom.labls import findlabl
from umpfea.mdl.build_k import ele8AG_calx


def calc_agap1(Rr, Rs, ecc, Nh_r, Nh_s, Ncirc=None, Nlayer=3):
    """
    Fourier-basis airgap stiffness at one eccentricity (along +x).

    Meshes the airgap with Nlayer x Ncirc 8-node quads, condenses to the
    rotor and stator boundary nodes and projects onto the harmonics.

    Parameters
    ----------
    Rr, Rs : float
        Rotor outer and stator inner radius (m).
    ecc : float
        Eccentricity along +x (m).
    Nh_r, Nh_s : int
        Highest rotor / stator harmonic.
    Ncirc : int or None
        Elements around the circumference (default max(200, 2*Nh_r, 2*Nh_s)).
    Nlayer : int
        Radial element layers.

    Returns
    -------
    Kcos : (Nh_r+Nh_s+2, Nh_r+Nh_s+2) ndarray
        DOFs [cos_rotor(0..Nh_r), cos_stator(0..Nh_s)]
    Ksin : (Nh_r+Nh_s, Nh_r+Nh_s) ndarray
        DOFs [sin_rotor(1..Nh_r), sin_stator(1..Nh_s)]
    """
    if Ncirc is None:
        Ncirc = max(200, Nh_r * 2, Nh_s * 2)
    else:
        if Ncirc < Nh_r:
            raise ValueError(f'calc_agap1: Ncirc={Ncirc} < Nh_r={Nh_r}')
        if Ncirc < Nh_s:
            raise ValueError(f'calc_agap1: Ncirc={Ncirc} < Nh_s={Nh_s}')

    if Rr > Rs:
        raise ValueError(
            f'calc_agap1: Rr={Rr*1000:.4f} mm > Rs={Rs*1000:.4f} mm')
    if ecc > (Rs - Rr):
        raise ValueError(
            f'calc_agap1: ecc={ecc*1000:.4f} mm > airgap={(Rs-Rr)*1000:.4f} mm')

    # Shift node angles with eccentricity so the elements stay well shaped
    frc   = 0.35
    Ncirc2 = 2 * Ncirc
    angls   = 2 * np.pi * np.arange(Ncirc2) / Ncirc2    # (Ncirc2,)
    angls_R = angls + (0.5 * ecc / Rr) * np.sin(angls)
    angls_S = angls - (0.5 * ecc / Rr) * np.sin(angls)
    for _ in range(10):
        angls_R = angls + (frc * ecc / Rr) * np.sin(angls_R)
        angls_S = angls - (frc * ecc / Rr) * np.sin(angls_S)

    # Nodes [label, x, y]. Odd node rows (element edges) have all Ncirc2
    # nodes, even rows (element mid-lines) only every other one.
    NL_off_layer = 2000 * int(np.ceil(Ncirc2 / 2000))
    NodSet_list = []

    for ilayer in range(1, 2 * Nlayer + 2):      # 1 .. 2*Nlayer+1
        NLoff  = (ilayer - 1) * NL_off_layer
        xstr   = (ilayer - 1) / (2 * Nlayer)     # stator fraction
        xrtr   = 1.0 - xstr                      # rotor fraction
        is_odd_r = ilayer % 2

        for icirc in range(1, Ncirc2 + 1):
            is_odd_c = icirc % 2
            if (is_odd_c + is_odd_r) > 0:
                labl  = NLoff + icirc
                ic0   = icirc - 1
                xy_r  = np.array([Rr * np.cos(angls_R[ic0]) + ecc,
                                   Rr * np.sin(angls_R[ic0])])
                xy_s  = np.array([Rs * np.cos(angls_S[ic0]),
                                   Rs * np.sin(angls_S[ic0])])
                xy_t  = xy_r * xrtr + xy_s * xstr
                NodSet_list.append([labl, xy_t[0], xy_t[1]])

    NodSet = np.array(NodSet_list)   # (NDoF, 3)
    NDoF   = NodSet.shape[0]

    label_to_idx = {int(NodSet[i, 0]): i for i in range(NDoF)}

    # Nlayer*Ncirc quad elements
    Elems4 = np.empty((Nlayer * Ncirc, 8), dtype=int)
    iele = 0
    for ilayer in range(1, Nlayer + 1):
        NLoff1 = ((ilayer - 1) * 2 + 0) * NL_off_layer
        NLoff2 = ((ilayer - 1) * 2 + 1) * NL_off_layer
        NLoff3 = ((ilayer - 1) * 2 + 2) * NL_off_layer
        topo_t = np.array([NLoff1 + 3, NLoff3 + 3, NLoff3 + 1, NLoff1 + 1,
                            NLoff2 + 3, NLoff3 + 2, NLoff2 + 1, NLoff1 + 2])
        for icirc in range(Ncirc):
            Elems4[iele, :] = topo_t
            iele += 1
            topo_t = topo_t + 2
        # Last element in the layer closes the ring
        topo_t -= 2
        topo_t[[0, 1, 4]] = [NLoff1 + 1, NLoff3 + 1, NLoff2 + 1]
        Elems4[iele - 1, :] = topo_t

    Nele = Nlayer * Ncirc

    k_rows, k_cols, k_vals = [], [], []

    for iele_idx in range(Nele):
        labls  = Elems4[iele_idx, :]
        indxs  = np.array([label_to_idx[int(l)] for l in labls])
        Node_X = NodSet[indxs, 1]
        Node_Y = NodSet[indxs, 2]
        k_e    = ele8AG_calx(Node_X, Node_Y)

        ii, jj = np.meshgrid(indxs, indxs, indexing='ij')
        k_rows.append(ii.ravel())
        k_cols.append(jj.ravel())
        k_vals.append(k_e.ravel())

    Kmat = sp.csr_matrix(
        (np.concatenate(k_vals),
         (np.concatenate(k_rows), np.concatenate(k_cols))),
        shape=(NDoF, NDoF))

    # Condense to the rotor ring (first Ncirc2 nodes) and the stator ring
    # (last Ncirc2 nodes)
    irng_inner = np.arange(0, Ncirc2)
    irng_outer = np.arange(NDoF - Ncirc2, NDoF)
    irng_m     = np.concatenate([irng_inner, irng_outer])
    irng_s     = np.arange(Ncirc2, NDoF - Ncirc2)

    K_mm = Kmat[np.ix_(irng_m, irng_m)]
    K_ms = Kmat[np.ix_(irng_m, irng_s)]
    K_sm = Kmat[np.ix_(irng_s, irng_m)]
    K_ss = Kmat[np.ix_(irng_s, irng_s)]

    lu   = spla.splu(K_ss.tocsc())
    T_sl = lu.solve(-K_sm.toarray())            # (NSl, NMs)
    Kred = K_mm.toarray() + K_ms.toarray() @ T_sl   # (4*Ncirc, 4*Ncirc)

    # Projection onto the harmonics, using each ring's own node angles
    T_cos_r = np.ones((Ncirc2, Nh_r + 1))   # col 0 = DC
    T_sin_r = np.zeros((Ncirc2, Nh_r))
    for ih in range(1, Nh_r + 1):
        T_cos_r[:, ih]      = np.cos(ih * angls_R)
        T_sin_r[:, ih - 1]  = np.sin(ih * angls_R)

    T_cos_s = np.ones((Ncirc2, Nh_s + 1))
    T_sin_s = np.zeros((Ncirc2, Nh_s))
    for ih in range(1, Nh_s + 1):
        T_cos_s[:, ih]      = np.cos(ih * angls_S)
        T_sin_s[:, ih - 1]  = np.sin(ih * angls_S)

    # Block-diagonal: rotor rows first, then stator rows
    Z_rs_c = np.zeros((Ncirc2, Nh_s + 1))
    Z_sr_c = np.zeros((Ncirc2, Nh_r + 1))
    Z_rs_s = np.zeros((Ncirc2, Nh_s))
    Z_sr_s = np.zeros((Ncirc2, Nh_r))

    T_cos2 = np.block([[T_cos_r, Z_rs_c],
                        [Z_sr_c, T_cos_s]])   # (4*Ncirc, Nh_r+Nh_s+2)
    T_sin2 = np.block([[T_sin_r, Z_rs_s],
                        [Z_sr_s, T_sin_s]])   # (4*Ncirc, Nh_r+Nh_s)

    Kcos = T_cos2.T @ Kred @ T_cos2           # (Nh_r+Nh_s+2, Nh_r+Nh_s+2)
    Ksin = T_sin2.T @ Kred @ T_sin2           # (Nh_r+Nh_s, Nh_r+Nh_s)

    return Kcos, Ksin


def calc_agapN(Rr, Rs, Nh_r, Nh_s, N_ecc=20, xmax=0.95, Ncirc=None, Nlayer=3):
    """
    Tabulate calc_agap1 over N_ecc eccentricities.

    The first three samples are closely spaced near zero; the rest are
    evenly spaced up to xmax*(Rs-Rr). Other arguments as in calc_agap1.

    Returns
    -------
    Eccs  : (N_ecc,) ndarray, sample eccentricities (m)
    KcosN : (Nh_r+Nh_s+2, Nh_r+Nh_s+2, N_ecc) ndarray
    KsinN : (Nh_r+Nh_s,   Nh_r+Nh_s,   N_ecc) ndarray
    """
    if Ncirc is None:
        Ncirc = max(200, Nh_r * 2, Nh_s * 2)

    P = Nh_r + Nh_s + 2
    Q = Nh_r + Nh_s
    KcosN = np.zeros((P, P, N_ecc))
    KsinN = np.zeros((Q, Q, N_ecc))

    airgap   = Rs - Rr
    Nom_Spac = (xmax * airgap) / (N_ecc - 1)
    xi       = Nom_Spac / 5
    Eccs     = np.zeros(N_ecc)
    Eccs[0:3] = np.array([0.0, 1.0, 2.0]) * xi / 2
    if N_ecc > 3:
        Eccs[3:N_ecc] = (xi
                         + (xmax * airgap - xi)
                         * np.arange(1, N_ecc - 2) / (N_ecc - 3))

    t_start = time.perf_counter()
    for i_ecc in range(N_ecc):
        Kcos, Ksin = calc_agap1(Rr, Rs, Eccs[i_ecc], Nh_r, Nh_s, Ncirc, Nlayer)
        KcosN[:, :, i_ecc] = Kcos
        KsinN[:, :, i_ecc] = Ksin
        elapsed = time.perf_counter() - t_start
        print(f' calc_agapN: ecc #{i_ecc+1:3d}/{N_ecc}'
              f'  ecc={Eccs[i_ecc]*1000:.4f} mm'
              f'  elapsed={elapsed:.1f}s')

    return Eccs, KcosN, KsinN


def calc_agapi(Eccs, KcosN, KsinN, NH_r, NH_s, ecc):
    """
    Interpolate Kcos, Ksin and their ecc-derivatives at eccentricity ecc.

    Inside the table, two overlapping quadratic fits are blended linearly;
    in the first and last intervals a single quadratic is used.

    Parameters
    ----------
    Eccs  : (N_ecc,) ndarray, sample eccentricities (m)
    KcosN : (P, P, N_ecc) ndarray
    KsinN : (Q, Q, N_ecc) ndarray
    NH_r, NH_s : int, rotor / stator harmonics
    ecc   : float, eccentricity (m), 0 <= ecc <= Eccs[-1]

    Returns
    -------
    Kcos       : (P, P) ndarray
    Ksin       : (Q, Q) ndarray
    d_Kcos_d_e : (P, P) ndarray
    d_Ksin_d_e : (Q, Q) ndarray
    """
    N_ecc = len(Eccs)

    if ecc > Eccs[-1]:
        raise ValueError(f'calc_agapi: ecc={ecc:.5f} > max(Eccs)={Eccs[-1]:.5f}')
    if ecc < 0.0:
        raise ValueError(f'calc_agapi: ecc={ecc:.5f} < 0')

    def _quad_wgts(ecc_3, tz_val):
        """Quadratic weights for value and d/d_ecc at normalised position tz_val."""
        tx0  = ecc_3[0]
        tx2  = ecc_3[2]
        xvec = (ecc_3 - tx0) / (tx2 - tx0)           # [0, x1, 1]
        Amat = np.column_stack([xvec**0, xvec**1, xvec**2])   # (3,3)
        bvec0 = np.array([1.0, tz_val,    tz_val**2])
        bvec1 = np.array([0.0, 1.0,    2*tz_val]) / (tx2 - tx0)
        wgts0 = np.linalg.solve(Amat.T, bvec0)
        wgts1 = np.linalg.solve(Amat.T, bvec1)
        return wgts0, wgts1

    def _apply_wgts(wgts, idxs):
        Kc = sum(wgts[k] * KcosN[:, :, idxs[k]] for k in range(len(idxs)))
        Ks = sum(wgts[k] * KsinN[:, :, idxs[k]] for k in range(len(idxs)))
        return Kc, Ks

    # First interval: quadratic through samples 0, 1, 2
    if ecc < Eccs[1]:
        tx0  = Eccs[0]
        tx2  = Eccs[2]
        tz   = (ecc - tx0) / (tx2 - tx0)
        w0, w1 = _quad_wgts(Eccs[0:3], tz)
        Kcos,       Ksin       = _apply_wgts(w0, [0, 1, 2])
        d_Kcos_d_e, d_Ksin_d_e = _apply_wgts(w1, [0, 1, 2])

    # Last interval: quadratic through the last three samples
    elif ecc >= Eccs[N_ecc - 2]:
        i0   = N_ecc - 3
        tx0  = Eccs[i0]
        tx2  = Eccs[N_ecc - 1]
        tz   = (ecc - tx0) / (tx2 - tx0)
        w0, w1 = _quad_wgts(Eccs[i0:N_ecc], tz)
        Kcos,       Ksin       = _apply_wgts(w0, [i0, i0+1, i0+2])
        d_Kcos_d_e, d_Ksin_d_e = _apply_wgts(w1, [i0, i0+1, i0+2])

    # Interior: blend quadratics through (i, j, k) and (j, k, l),
    # where Eccs[j] <= ecc < Eccs[k]
    else:
        j_ec = int(np.where(Eccs <= ecc)[0][-1])
        i_ec = j_ec - 1
        k_ec = j_ec + 1
        l_ec = j_ec + 2

        tx0_ijk = Eccs[i_ec]
        tx2_ijk = Eccs[k_ec]
        tz_ijk  = (ecc - tx0_ijk) / (tx2_ijk - tx0_ijk)
        w0_ijk, w1_ijk = _quad_wgts(Eccs[i_ec:k_ec+1], tz_ijk)

        tx0_jkl = Eccs[j_ec]
        tx2_jkl = Eccs[l_ec]
        tz_jkl  = (ecc - tx0_jkl) / (tx2_jkl - tx0_jkl)
        w0_jkl, w1_jkl = _quad_wgts(Eccs[j_ec:l_ec+1], tz_jkl)

        at = (Eccs[k_ec] - ecc) / (Eccs[k_ec] - Eccs[j_ec])
        bt = (ecc - Eccs[j_ec]) / (Eccs[k_ec] - Eccs[j_ec])

        # Weights on samples i, j, k, l
        w0 = at * np.array([w0_ijk[0], w0_ijk[1], w0_ijk[2], 0.0]) \
           + bt * np.array([0.0, w0_jkl[0], w0_jkl[1], w0_jkl[2]])
        w1 = at * np.array([w1_ijk[0], w1_ijk[1], w1_ijk[2], 0.0]) \
           + bt * np.array([0.0, w1_jkl[0], w1_jkl[1], w1_jkl[2]])

        Kcos       = (w0[0]*KcosN[:,:,i_ec] + w0[1]*KcosN[:,:,j_ec]
                    + w0[2]*KcosN[:,:,k_ec] + w0[3]*KcosN[:,:,l_ec])
        Ksin       = (w0[0]*KsinN[:,:,i_ec] + w0[1]*KsinN[:,:,j_ec]
                    + w0[2]*KsinN[:,:,k_ec] + w0[3]*KsinN[:,:,l_ec])
        d_Kcos_d_e = (w1[0]*KcosN[:,:,i_ec] + w1[1]*KcosN[:,:,j_ec]
                    + w1[2]*KcosN[:,:,k_ec] + w1[3]*KcosN[:,:,l_ec])
        d_Ksin_d_e = (w1[0]*KsinN[:,:,i_ec] + w1[1]*KsinN[:,:,j_ec]
                    + w1[2]*KsinN[:,:,k_ec] + w1[3]*KsinN[:,:,l_ec])

    return Kcos, Ksin, d_Kcos_d_e, d_Ksin_d_e


def calc_agapj(Kcos, Ksin, d_Kc_d_e, d_Ks_d_e, NH_r, NH_s, q_ecc, q_r):
    """
    Airgap stiffness K_AG at eccentricity angle q_ecc and rotor angle q_r,
    with its derivatives.

    K_AG DOF order (size 2*(NH_r+NH_s+1)):
      [cos_rotor(0..NH_r), sin_rotor(1..NH_r),
       cos_stator(0..NH_s), sin_stator(1..NH_s)]
    Rotor harmonics are in the rotor frame, stator harmonics in the
    stator frame.

    Parameters
    ----------
    Kcos, Ksin         : from calc_agapi (eccentricity along +x)
    d_Kc_d_e, d_Ks_d_e : their ecc-derivatives
    NH_r, NH_s         : int, rotor / stator harmonics
    q_ecc              : float, eccentricity angle from +x (rad)
    q_r                : float, rotor angle (rad)

    Returns
    -------
    K_AG, d_KAG_d_ecc, d_KAG_d_qe, d_KAG_d_qr, d_KAG_d_n
        K_AG and its derivatives w.r.t. ecc, q_ecc, q_r, and (for use at
        ecc = 0) displacement normal to the eccentricity direction.
    """
    NDoF_AG     = 2 * (NH_r + NH_s + 1)
    K_AG        = np.zeros((NDoF_AG, NDoF_AG))
    d_KAG_d_ecc = np.zeros((NDoF_AG, NDoF_AG))
    d_KAG_d_qe  = np.zeros((NDoF_AG, NDoF_AG))
    d_KAG_d_qr  = np.zeros((NDoF_AG, NDoF_AG))
    d_KAG_d_n   = np.zeros((NDoF_AG, NDoF_AG))

    # Blocks of K_AG
    irngF1 = np.arange(0,         NH_r + 1)             # cos_rotor (NH_r+1)
    irngF2 = np.arange(NH_r + 1,  2*NH_r + 1)           # sin_rotor (NH_r)
    irngF3 = np.arange(2*NH_r+1,  2*NH_r + NH_s + 2)    # cos_stator (NH_s+1)
    irngF4 = np.arange(2*NH_r+NH_s+2, NDoF_AG)          # sin_stator (NH_s)

    # Matching ranges in Kcos / Ksin
    irngCr = np.arange(0,         NH_r + 1)              # cos_rotor in Kcos
    irngCs = np.arange(NH_r + 1,  NH_r + NH_s + 2)       # cos_stator in Kcos
    irngSr = np.arange(0,         NH_r)                  # sin_rotor in Ksin
    irngSs = np.arange(NH_r,      NH_r + NH_s)           # sin_stator in Ksin

    # Fill the unrotated blocks
    K_AG[np.ix_(irngF1, irngF1)] = Kcos[np.ix_(irngCr, irngCr)]
    K_AG[np.ix_(irngF2, irngF2)] = Ksin[np.ix_(irngSr, irngSr)]
    K_AG[np.ix_(irngF3, irngF3)] = Kcos[np.ix_(irngCs, irngCs)]
    K_AG[np.ix_(irngF4, irngF4)] = Ksin[np.ix_(irngSs, irngSs)]
    K_AG[np.ix_(irngF1, irngF3)] = Kcos[np.ix_(irngCr, irngCs)]
    K_AG[np.ix_(irngF2, irngF4)] = Ksin[np.ix_(irngSr, irngSs)]
    K_AG[np.ix_(irngF3, irngF1)] = Kcos[np.ix_(irngCs, irngCr)]
    K_AG[np.ix_(irngF4, irngF2)] = Ksin[np.ix_(irngSs, irngSr)]

    d_KAG_d_ecc[np.ix_(irngF1, irngF1)] = d_Kc_d_e[np.ix_(irngCr, irngCr)]
    d_KAG_d_ecc[np.ix_(irngF2, irngF2)] = d_Ks_d_e[np.ix_(irngSr, irngSr)]
    d_KAG_d_ecc[np.ix_(irngF3, irngF3)] = d_Kc_d_e[np.ix_(irngCs, irngCs)]
    d_KAG_d_ecc[np.ix_(irngF4, irngF4)] = d_Ks_d_e[np.ix_(irngSs, irngSs)]
    d_KAG_d_ecc[np.ix_(irngF1, irngF3)] = d_Kc_d_e[np.ix_(irngCr, irngCs)]
    d_KAG_d_ecc[np.ix_(irngF2, irngF4)] = d_Ks_d_e[np.ix_(irngSr, irngSs)]
    d_KAG_d_ecc[np.ix_(irngF3, irngF1)] = d_Kc_d_e[np.ix_(irngCs, irngCr)]
    d_KAG_d_ecc[np.ix_(irngF4, irngF2)] = d_Ks_d_e[np.ix_(irngSs, irngSr)]

    # d_KAG_d_n (ecc = 0 only) is built from d_KAG_d_ecc with sign masks
    # such as cos(i*pi/2)*sin(j*pi/2), rounded to exact 0/+-1
    def _mask(row_fn, col_fn, nrows, ncols):
        r = row_fn(np.arange(nrows))
        c = col_fn(np.arange(ncols))
        return np.round(np.outer(r, c))

    cos_r = lambda v: np.cos(np.pi / 2 * v)
    sin_r = lambda v: np.sin(np.pi / 2 * v)

    Mask_Rc_Rs = _mask(lambda v: cos_r(v),         # rows n = 0..NH_r
                       lambda v: sin_r(v + 1),      # cols n = 1..NH_r
                       NH_r + 1, NH_r)
    Mask_Rs_Rc = _mask(lambda v: sin_r(v + 1),
                       lambda v: cos_r(v),
                       NH_r, NH_r + 1)
    Mask_Rc_Ss = _mask(lambda v: cos_r(v),
                       lambda v: sin_r(v + 1),
                       NH_r + 1, NH_s)
    Mask_Rs_Sc = _mask(lambda v: sin_r(v + 1),
                       lambda v: cos_r(v),
                       NH_r, NH_s + 1)
    Mask_Sc_Rs = _mask(lambda v: cos_r(v),
                       lambda v: sin_r(v + 1),
                       NH_s + 1, NH_r)
    Mask_Ss_Rc = _mask(lambda v: sin_r(v + 1),
                       lambda v: cos_r(v),
                       NH_s, NH_r + 1)
    Mask_Sc_Ss = _mask(lambda v: cos_r(v),
                       lambda v: sin_r(v + 1),
                       NH_s + 1, NH_s)
    Mask_Ss_Sc = _mask(lambda v: sin_r(v + 1),
                       lambda v: cos_r(v),
                       NH_s, NH_s + 1)

    ir1  = irngF1
    ir2  = irngF2
    ir3  = irngF3
    ir4  = irngF4
    jr1  = irngF1[1:]    # cos_rotor without DC
    jr3  = irngF3[1:]    # cos_stator without DC

    # Block (1,2)
    d_KAG_d_n[np.ix_(ir1, ir2)]  = d_KAG_d_ecc[np.ix_(ir1, jr1)] * Mask_Rc_Rs
    d_KAG_d_n[np.ix_(jr1, ir2)] -= d_KAG_d_ecc[np.ix_(ir2, ir2)] * Mask_Rs_Rc[:, 1:]
    # Block (1,4)
    d_KAG_d_n[np.ix_(ir1, ir4)]  = d_KAG_d_ecc[np.ix_(ir1, jr3)] * Mask_Rc_Ss
    d_KAG_d_n[np.ix_(jr1, ir4)] -= d_KAG_d_ecc[np.ix_(ir2, ir4)] * Mask_Rs_Sc[:, 1:]
    # Block (2,1)
    d_KAG_d_n[np.ix_(ir2, ir1)]  = d_KAG_d_ecc[np.ix_(jr1, ir1)] * Mask_Rs_Rc
    d_KAG_d_n[np.ix_(ir2, jr1)] -= d_KAG_d_ecc[np.ix_(ir2, ir2)] * Mask_Rc_Rs[1:, :]
    # Block (2,3)
    d_KAG_d_n[np.ix_(ir2, ir3)]  = d_KAG_d_ecc[np.ix_(jr1, ir3)] * Mask_Rs_Sc
    d_KAG_d_n[np.ix_(ir2, jr3)] -= d_KAG_d_ecc[np.ix_(ir2, ir4)] * Mask_Rc_Ss[1:, :]
    # Block (3,2)
    d_KAG_d_n[np.ix_(ir3, ir2)]  = d_KAG_d_ecc[np.ix_(ir3, jr1)] * Mask_Sc_Rs
    d_KAG_d_n[np.ix_(jr3, ir2)] -= d_KAG_d_ecc[np.ix_(ir4, ir2)] * Mask_Ss_Rc[:, 1:]
    # Block (3,4)
    d_KAG_d_n[np.ix_(ir3, ir4)]  = d_KAG_d_ecc[np.ix_(ir3, jr3)] * Mask_Sc_Ss
    d_KAG_d_n[np.ix_(jr3, ir4)] -= d_KAG_d_ecc[np.ix_(ir4, ir4)] * Mask_Ss_Sc[:, 1:]
    # Block (4,1)
    d_KAG_d_n[np.ix_(ir4, ir1)]  = d_KAG_d_ecc[np.ix_(jr3, ir1)] * Mask_Ss_Rc
    d_KAG_d_n[np.ix_(ir4, jr1)] -= d_KAG_d_ecc[np.ix_(ir4, ir2)] * Mask_Sc_Rs[1:, :]
    # Block (4,3)
    d_KAG_d_n[np.ix_(ir4, ir3)]  = d_KAG_d_ecc[np.ix_(jr3, ir3)] * Mask_Ss_Sc
    d_KAG_d_n[np.ix_(ir4, jr3)] -= d_KAG_d_ecc[np.ix_(ir4, ir4)] * Mask_Sc_Ss[1:, :]

    # Rotate the rotor harmonics by (q_ecc - q_r): Kcos/Ksin were built with
    # the eccentricity along +x
    S_2x2 = np.array([[0.0, -1.0], [1.0, 0.0]])

    qq_r = q_ecc - q_r
    qq_r -= 2 * np.pi * round(qq_r / (2 * np.pi))

    for iH_r in range(1, NH_r + 1):
        irng = np.array([iH_r, NH_r + iH_r])   # [cos, sin] of harmonic iH_r
        iqqr = qq_r * iH_r
        c, s = np.cos(iqqr), np.sin(iqqr)
        R2x2 = np.array([[c, -s], [s, c]])

        K_AG[irng, :]        = R2x2 @ K_AG[irng, :]
        K_AG[:, irng]        = K_AG[:, irng] @ R2x2.T
        d_KAG_d_ecc[irng, :] = R2x2 @ d_KAG_d_ecc[irng, :]
        d_KAG_d_ecc[:, irng] = d_KAG_d_ecc[:, irng] @ R2x2.T
        d_KAG_d_n[irng, :]   = R2x2 @ d_KAG_d_n[irng, :]
        d_KAG_d_n[:, irng]   = d_KAG_d_n[:, irng] @ R2x2.T

    # Rotate the stator harmonics by q_ecc
    for iH_s in range(1, NH_s + 1):
        irng = np.array([2*NH_r + 1 + iH_s,
                         2*NH_r + NH_s + 1 + iH_s])
        iqqs = q_ecc * iH_s
        c, s = np.cos(iqqs), np.sin(iqqs)
        R2x2 = np.array([[c, -s], [s, c]])

        K_AG[irng, :]        = R2x2 @ K_AG[irng, :]
        K_AG[:, irng]        = K_AG[:, irng] @ R2x2.T
        d_KAG_d_ecc[irng, :] = R2x2 @ d_KAG_d_ecc[irng, :]
        d_KAG_d_ecc[:, irng] = d_KAG_d_ecc[:, irng] @ R2x2.T
        d_KAG_d_n[irng, :]   = R2x2 @ d_KAG_d_n[irng, :]
        d_KAG_d_n[:, irng]   = d_KAG_d_n[:, irng] @ R2x2.T

    # Angular derivatives from the rotated K_AG:
    #   d/dq (R K R') = iH * (S K_rot + K_rot S'),  R = R(iH*q)
    # q_ecc rotates rotor and stator harmonics; q_r only the rotor, with
    # the opposite sign.
    for iH_r in range(1, NH_r + 1):
        irng = np.array([iH_r, NH_r + iH_r])
        d_KAG_d_qe[irng, :]  += iH_r * (S_2x2 @ K_AG[irng, :])
        d_KAG_d_qe[:, irng]  += iH_r * (K_AG[:, irng] @ S_2x2.T)
        d_KAG_d_qr[irng, :]  -= iH_r * (S_2x2 @ K_AG[irng, :])
        d_KAG_d_qr[:, irng]  -= iH_r * (K_AG[:, irng] @ S_2x2.T)

    for iH_s in range(1, NH_s + 1):
        irng = np.array([2*NH_r + 1 + iH_s,
                         2*NH_r + NH_s + 1 + iH_s])
        d_KAG_d_qe[irng, :]  += iH_s * (S_2x2 @ K_AG[irng, :])
        d_KAG_d_qe[:, irng]  += iH_s * (K_AG[:, irng] @ S_2x2.T)

    return K_AG, d_KAG_d_ecc, d_KAG_d_qe, d_KAG_d_qr, d_KAG_d_n
