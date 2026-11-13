import numpy as np
from numba import njit
from umpfea.geom.labls import findlabl
from umpfea.mdl.build_k import eval_6_sfs, eval_8_sfs, ele6_area, ele8_area


def bender(ndiv, rat):
    """
    ndiv+1 points on [-1, 1], with the last interval about `rat` times the
    first (quadratic grading).
    """
    a = (rat - 1) / (1 + rat)
    b = 1 - a
    xdivs = np.arange(ndiv + 1) / ndiv
    ydivs = a * xdivs**2 + b * xdivs
    return 2.0 * ydivs - 1.0


def gen_SMR4(KeyPts, KP_Lbls, DscrInsts, DI_Lbls, N_Lbl_O, PD_Lbl, M_Lbl, CI_Lbl):
    """
    Structured mesh of 8-node quads over a region bounded by 8 KeyPoints.

    KeyPoint layout and (ksi, eta):
        2  6  3        1 (-1,-1)  2 (-1, 1)  3 (1, 1)  4 (1,-1)
        5     7        5 (-1, 0)  6 ( 0, 1)  7 (1, 0)  8 (0,-1)
        1  8  4

    Parameters
    ----------
    KeyPts    : (K, 3+) ndarray, [label, x, y, ...]
    KP_Lbls   : the 8 KeyPoint labels, in the order above
    DscrInsts : discretisation table, [label, N_divs, ratio]
    DI_Lbls   : 2 labels; DI_Lbls[0] sets the eta direction and DI_Lbls[1]
                the ksi direction. A negative label inverts the ratio.
    N_Lbl_O   : node label offset; new nodes are N_Lbl_O+1, ...
    PD_Lbl, M_Lbl, CI_Lbl : PDir, material and CMap labels for all elements

    Returns
    -------
    NodeSet_ref : (n_nodes, 3) ndarray, [label, ksi, eta]
    NodeSet_glb : (n_nodes, 3) ndarray, [label, x, y]
    Elems4      : (N1*N2, 11) ndarray
    Bound       : (2*(N1+N2)*2, 2) ndarray, boundary segments, anticlockwise
    """
    DI_Lbls = np.asarray(DI_Lbls).ravel()

    # DI_Lbls[0] -> eta (N2 divisions), DI_Lbls[1] -> ksi (N1 divisions)
    DI_Indxs = findlabl(DscrInsts, np.abs(DI_Lbls)).ravel()
    if DI_Indxs.min() < 0:
        raise ValueError(
            f"ERROR in gen_SMR4: one or more DI labels not found.\n"
            f"Labels: {DI_Lbls}\nIndices: {DI_Indxs}"
        )

    N2 = int(DscrInsts[DI_Indxs[0], 1])
    X2 = float(DscrInsts[DI_Indxs[0], 2])
    if DI_Lbls[0] < 0:
        X2 = 1.0 / X2

    N1 = int(DscrInsts[DI_Indxs[1], 1])
    X1 = float(DscrInsts[DI_Indxs[1], 2])
    if DI_Lbls[1] < 0:
        X1 = 1.0 / X1

    # Corner and mid-side positions along each direction
    ksi_array = bender(2 * N1, X1)
    eta_array = bender(2 * N2, X2)

    KP_Lbls  = np.asarray(KP_Lbls).ravel()
    KP_Indxs = findlabl(KeyPts, KP_Lbls).ravel()
    if KP_Indxs.min() < 0:
        raise ValueError(
            f"ERROR in gen_SMR4: one or more KP labels not found.\n"
            f"Labels: {KP_Lbls}\nIndices: {KP_Indxs}"
        )

    xv_kpts = KeyPts[KP_Indxs, 1]
    yv_kpts = KeyPts[KP_Indxs, 2]

    # Full (2*N1+1) x (2*N2+1) grid, mapped through the 8-node shape functions
    n_ksi = len(ksi_array)
    n_eta = len(eta_array)
    n_all = n_ksi * n_eta

    NodeSet_ref = np.zeros((n_all, 3))
    NodeSet_glb = np.zeros((n_all, 3))

    kk = 0
    for i_eta in range(n_eta):
        eta_t = eta_array[i_eta]
        for i_ksi in range(n_ksi):
            ksi_t  = ksi_array[i_ksi]
            lbl    = kk + 1 + N_Lbl_O
            t_vec8 = eval_8_sfs(ksi_t, eta_t, 0)
            x_t    = t_vec8 @ xv_kpts
            y_t    = t_vec8 @ yv_kpts
            NodeSet_ref[kk, :] = [lbl, ksi_t, eta_t]
            NodeSet_glb[kk, :] = [lbl, x_t,   y_t  ]
            kk += 1

    Elems4 = np.zeros((N1 * N2, 11))
    iele = 0
    for i_eta in range(N2):
        t_labl = N_Lbl_O + 1 + i_eta * 2 * n_ksi
        topo   = np.array([
            t_labl,
            t_labl + 2 * n_ksi,
            t_labl + 2 * n_ksi + 2,
            t_labl + 2,
            t_labl + n_ksi,
            t_labl + 2 * n_ksi + 1,
            t_labl + n_ksi + 2,
            t_labl + 1,
        ], dtype=float)
        for i_ksi in range(N1):
            Elems4[iele, :] = np.concatenate([topo, [PD_Lbl, M_Lbl, CI_Lbl]])
            topo  += 2
            iele  += 1

    # Boundary segments, walking the four sides in turn
    rows = []
    k = N_Lbl_O + 1
    for _ in range(N1 * 2):        # ksi = -1 .. 1 at eta = -1
        j = k + 1
        rows.append([k, j])
        k = j
    for _ in range(N2 * 2):        # eta = -1 .. 1 at ksi = 1
        j = k + n_ksi
        rows.append([k, j])
        k = j
    for _ in range(N1 * 2):        # ksi = 1 .. -1 at eta = 1
        j = k - 1
        rows.append([k, j])
        k = j
    for _ in range(N2 * 2):        # eta = 1 .. -1 at ksi = -1
        j = k - n_ksi
        rows.append([k, j])
        k = j
    Bound = np.array(rows, dtype=float)

    # Drop the element centre nodes (not used by serendipity elements)
    flags = np.zeros(n_all, dtype=int)
    for i1 in range(1, N2 + 1):
        idxt = (2 * i1 - 1) * n_ksi    # 1-based, hence idxt - 1 below
        for i2 in range(N1):
            idxt += 2
            flags[idxt - 1] = 1

    indcs_reds  = np.where(flags == 1)[0]
    NodeSet_ref = np.delete(NodeSet_ref, indcs_reds, axis=0)
    NodeSet_glb = np.delete(NodeSet_glb, indcs_reds, axis=0)

    return NodeSet_ref, NodeSet_glb, Elems4, Bound


@njit(cache=True)
def _find_cnodes_kernel(coords, labels, tol_sq):
    """Numba kernel for find_cnodes: count the close pairs, then fill them."""
    NN = coords.shape[0]

    count = 0
    for i in range(NN):
        xi = coords[i, 0]
        yi = coords[i, 1]
        for j in range(i + 1, NN):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            if dx * dx + dy * dy < tol_sq:
                count += 1

    result = np.zeros((count, 2), dtype=np.float64)

    k = 0
    for i in range(NN):
        xi = coords[i, 0]
        yi = coords[i, 1]
        for j in range(i + 1, NN):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            if dx * dx + dy * dy < tol_sq:
                result[k, 0] = labels[i]
                result[k, 1] = labels[j]
                k += 1

    return result


def find_cnodes(NodeSet, tol):
    """
    Pairs of nodes closer than tol.

    Returns
    -------
    (n_pairs, 2) ndarray of labels [li, lj]; li is kept, lj is replaced.
    """
    labels = np.round(NodeSet[:, 0])
    coords = np.ascontiguousarray(NodeSet[:, 1:3])
    result = _find_cnodes_kernel(coords, labels, tol * tol)

    if result.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    return result


def kill_cnodes(EleSet0, ntopo, Bound0, cnodes):
    """
    Replace lj by li for each pair in cnodes, in the element topology and
    boundary arrays.

    Parameters
    ----------
    EleSet0 : (Nele, ncols) ndarray; the first ntopo columns are node labels
    ntopo   : int, 6 for Elems3, 8 for Elems4
    Bound0  : (Nbnd, 2) ndarray
    cnodes  : (n_pairs, 2) ndarray from find_cnodes

    Returns
    -------
    EleSet1, Bound1 : updated copies
    """
    EleSet1 = EleSet0.copy()
    Bound1  = Bound0.copy()

    for pair in cnodes:
        li = pair[0]
        lj = pair[1]
        mask_e = EleSet1[:, :ntopo] == lj
        EleSet1[:, :ntopo][mask_e] = li
        mask_b = Bound1 == lj
        Bound1[mask_b] = li

    return EleSet1, Bound1


def find_tnodes4(NodeSet, Elems4):
    """Flag array, 1 for each NodeSet row used by an Elems4 element."""
    nnod            = NodeSet.shape[0]
    topo_nodes_flgs = np.zeros(nnod, dtype=int)
    Nele            = Elems4.shape[0]

    for iele in range(Nele):
        labls = Elems4[iele, :8]
        indxs = findlabl(NodeSet, labls).ravel()
        if indxs.min() < 0:
            raise ValueError(
                f"ERROR in find_tnodes4: negative index for element {iele}.\n"
                f"Labels: {labls}\nIndices: {indxs}"
            )
        topo_nodes_flgs[indxs] = 1

    return topo_nodes_flgs


def cncl_bsegs(bound):
    """
    Remove boundary segments that appear in both directions, e.g. [a, b] and
    [b, a] (internal edges between regions).
    """
    nbnd  = bound.shape[0]
    flags = np.ones(nbnd, dtype=int)    # 0 = cancelled

    for ibnd in range(nbnd):
        if flags[ibnd] == 0:
            continue
        it = bound[ibnd, 0]
        jt = bound[ibnd, 1]
        for jbnd in range(ibnd + 1, nbnd):
            if flags[jbnd] == 0:
                continue
            if bound[jbnd, 0] == jt and bound[jbnd, 1] == it:
                flags[ibnd] = 0
                flags[jbnd] = 0
                break

    indxs = np.where(flags == 1)[0]
    return bound[indxs, :]


def gen_SMR4_set(KeyPts, DscrInsts, SMR4set, N_Lbl_O=0):
    """
    Mesh several SMR4 regions and merge the nodes on shared edges.

    Parameters
    ----------
    KeyPts    : (K, 3+) ndarray, [label, x, y, ...]
    DscrInsts : discretisation table
    SMR4set   : (nSMR4, 13) ndarray, one row per region:
                  0-7  KeyPoint labels
                  8-9  DI labels
                  10   PDir label
                  11   material label
                  12   CMap label
    N_Lbl_O   : int, node label offset

    Returns
    -------
    NodSet_x : (n_nodes, 3) ndarray, [label, x, y]
    Elems4_x : (n_elems, 11) ndarray
    Bound_x  : (n_bsegs, 2) ndarray
    """
    N_Label_Off = N_Lbl_O
    nSMR4   = SMR4set.shape[0]

    NodSet_y = np.zeros((0, 3))
    Elems4_y = np.zeros((0, 11))
    Bound_y  = np.zeros((0, 2))

    for iSMR4 in range(nSMR4):
        SMR4t = SMR4set[iSMR4, :]
        _, NodSet, Elems4, Bound = gen_SMR4(
            KeyPts, SMR4t[0:8], DscrInsts, SMR4t[8:10],
            N_Label_Off, SMR4t[10], SMR4t[11], SMR4t[12]
        )
        N_Label_Off = int(NodSet[:, 0].max())
        NodSet_y = np.vstack([NodSet_y, NodSet])
        Elems4_y = np.vstack([Elems4_y, Elems4])
        Bound_y  = np.vstack([Bound_y,  Bound ])

    # Merge coincident nodes on shared edges
    max_x = np.abs(NodSet_y[:, 1]).max()
    max_y = np.abs(NodSet_y[:, 2]).max()
    tol   = 1e-12 * np.sqrt(max_x**2 + max_y**2)
    cnods = find_cnodes(NodSet_y, tol)

    Elems4_x, Bound_z = kill_cnodes(Elems4_y, 8, Bound_y, cnods)

    topo_flags          = find_tnodes4(NodSet_y, Elems4_x)
    dead_node_indices   = np.where(topo_flags == 0)[0]
    NodSet_x            = np.delete(NodSet_y, dead_node_indices, axis=0)

    Bound_x = cncl_bsegs(Bound_z)

    print(f"\n Function gen_SMR4_set completed:\n")
    print(f" Total nodes generated:              {NodSet_y.shape[0]:4d}")
    print(f" Coincident node pairs found:        {cnods.shape[0]:4d}")
    print(f" Final nodes retained:               {NodSet_x.shape[0]:4d}")
    print(f" Final boundary segments:            {Bound_x.shape[0]:4d}")

    return NodSet_x, Elems4_x, Bound_x


# SMR3: triangular regions

def find_tnodes3(NodeSet, Elems3):
    """Flag array, 1 for each NodeSet row used by an Elems3 element."""
    nnod            = NodeSet.shape[0]
    topo_nodes_flgs = np.zeros(nnod, dtype=int)
    Nele            = Elems3.shape[0]

    for iele in range(Nele):
        labls = Elems3[iele, :6]
        indxs = findlabl(NodeSet, labls).ravel()
        if indxs.min() < 0:
            raise ValueError(
                f"ERROR in find_tnodes3: negative index for element {iele}.\n"
                f"Labels: {labls}\nIndices: {indxs}"
            )
        topo_nodes_flgs[indxs] = 1

    return topo_nodes_flgs


def gen_SMR3(KeyPts, KP_Lbls, DscrInsts, DI_Lbls, N_Lbl_O, PD_Lbl, M_Lbl, CI_Lbl):
    """
    Structured mesh over a triangular region bounded by 6 KeyPoints:
    8-node quads, with one row of 6-node triangles at the apex (KP1).

    KeyPoint layout and (ksi, eta):
        2              1 (-1,-1)  2 (-1, 1)  3 (1,-1)
        4  5           4 (-1, 0)  5 ( 0, 0)  6 (0,-1)
        1  6  3

    A (zeta, theta) grid is collapsed onto the apex: zeta runs from the
    KP2-KP3 base (0) to KP1 (1). DI_Lbls[0] sets the zeta divisions (N1),
    DI_Lbls[1] those along the base (N2). A negative label inverts the ratio.
    Other arguments as in gen_SMR4.

    Returns
    -------
    NodeSet_ref : (n_nodes, 3) ndarray, [label, ksi, eta]
    NodeSet_glb : (n_nodes, 3) ndarray, [label, x, y]
    Elems3      : (N2, 9) ndarray
    Elems4      : ((N1-1)*N2, 11) ndarray
    Bound       : boundary segments
    """
    DI_Lbls = np.asarray(DI_Lbls).ravel()

    DI_Indxs = findlabl(DscrInsts, np.abs(DI_Lbls)).ravel()
    if DI_Indxs.min() < 0:
        raise ValueError(
            f"ERROR in gen_SMR3: one or more DI labels not found.\n"
            f"Labels: {DI_Lbls}\nIndices: {DI_Indxs}"
        )

    N1 = int(DscrInsts[DI_Indxs[0], 1])
    X1 = float(DscrInsts[DI_Indxs[0], 2])
    if DI_Lbls[0] < 0:
        X1 = 1.0 / X1

    N2 = int(DscrInsts[DI_Indxs[1], 1])
    X2 = float(DscrInsts[DI_Indxs[1], 2])
    if DI_Lbls[1] < 0:
        X2 = 1.0 / X2

    KP_Lbls  = np.asarray(KP_Lbls).ravel()
    KP_Indxs = findlabl(KeyPts, KP_Lbls).ravel()
    if KP_Indxs.min() < 0:
        raise ValueError(
            f"ERROR in gen_SMR3: one or more KP labels not found.\n"
            f"Labels: {KP_Lbls}\nIndices: {KP_Indxs}"
        )

    xv_kpts = KeyPts[KP_Indxs, 1]
    yv_kpts = KeyPts[KP_Indxs, 2]

    zeta_array = bender(2 * N1, X1)
    zeta_array = (zeta_array + 1) / 2    # [-1, 1] -> [0, 1]
    nz = 2 * N1 + 1

    # theta runs the other way to zeta, hence 1/X2 and the sign flip below
    thet_array = bender(2 * N2, 1.0 / X2)
    nt = 2 * N2 + 1

    n_all       = nz * nt
    NodeSet_ref = np.zeros((n_all, 3))
    NodeSet_glb = np.zeros((n_all, 3))

    kk = 0
    for iz in range(nz):
        ztt = zeta_array[iz]
        for it in range(nt):
            ttt    = -thet_array[it]
            ksi_tt = -ztt + (1 - ztt) * ttt
            eta_tt = -ztt - (1 - ztt) * ttt
            lbl    = kk + 1 + N_Lbl_O
            t_vec6 = eval_6_sfs(ksi_tt, eta_tt, 0)
            x_t    = t_vec6 @ xv_kpts
            y_t    = t_vec6 @ yv_kpts
            NodeSet_ref[kk, :] = [lbl, ksi_tt, eta_tt]
            NodeSet_glb[kk, :] = [lbl, x_t,    y_t   ]
            kk += 1

    # Quads: all rows except the last
    Elems4 = np.zeros(((N1 - 1) * N2, 11))
    iele = 0
    for i1 in range(N1 - 1):
        t_labl = N_Lbl_O + 1 + i1 * 2 * nt
        topo   = np.array([
            t_labl,
            t_labl + 2 * nt,
            t_labl + 2 * nt + 2,
            t_labl + 2,
            t_labl + nt,
            t_labl + 2 * nt + 1,
            t_labl + nt + 2,
            t_labl + 1,
        ], dtype=float)
        for i2 in range(N2):
            Elems4[iele, :] = np.concatenate([topo, [PD_Lbl, M_Lbl, CI_Lbl]])
            topo  += 2
            iele  += 1

    # Triangles: last row, all sharing the apex node
    Elems3 = np.zeros((N2, 9))
    tip_labl = N_Lbl_O + 1 + N1 * 2 * nt
    t_labl   = N_Lbl_O + 1 + (N1 - 1) * 2 * nt
    for i2 in range(N2):
        topo = np.array([
            t_labl,
            tip_labl,
            t_labl + 2,
            t_labl + nt,
            t_labl + nt + 2,
            t_labl + 1,
        ], dtype=float)
        Elems3[i2, :] = np.concatenate([topo, [PD_Lbl, M_Lbl, CI_Lbl]])
        t_labl += 2

    # Boundary segments
    rows = []

    # Base, KP3 to KP2
    k = N_Lbl_O + 1
    for _ in range(N2 * 2):
        j = k + 1
        rows.append([k, j])
        k = j

    # KP2 to KP1; the last step goes to the single apex node
    for _ in range(N1 * 2 - 1):
        j = k + nt
        rows.append([k, j])
        k = j
    j = N_Lbl_O + nt * (nz - 1) + 1
    rows.append([k, j])
    k = j

    # KP1 back to KP3
    for _ in range(N1 * 2):
        j = k - nt
        rows.append([k, j])
        k = j

    Bound = np.array(rows, dtype=float)

    # Drop element centre nodes (as in gen_SMR4) ...
    flags = np.zeros(n_all, dtype=int)
    for i1 in range(1, N1 + 1):
        idxt = (2 * i1 - 1) * nt    # 1-based, hence idxt - 1 below
        for i2 in range(N2):
            idxt += 2
            flags[idxt - 1] = 1

    # ... and the duplicate apex nodes (last row except its first node)
    irng = np.arange((nz - 1) * nt + 1, nz * nt)
    flags[irng] += 1

    indcs_reds  = np.where(flags == 1)[0]
    NodeSet_ref = np.delete(NodeSet_ref, indcs_reds, axis=0)
    NodeSet_glb = np.delete(NodeSet_glb, indcs_reds, axis=0)

    return NodeSet_ref, NodeSet_glb, Elems3, Elems4, Bound


def gen_SMR3_set(KeyPts, DscrInsts, SMR3set, N_Lbl_O=0):
    """
    Mesh several SMR3 regions and merge the nodes on shared edges.

    SMR3set rows: 0-5 KeyPoint labels, 6-7 DI labels, 8 PDir, 9 material,
    10 CMap. Otherwise as gen_SMR4_set.

    Returns
    -------
    NodSet_x : (n_nodes, 3) ndarray, [label, x, y]
    Elems3_x : (n_tri, 9) ndarray
    Elems4_x : (n_quad, 11) ndarray
    Bound_x  : (n_bsegs, 2) ndarray
    """
    N_Label_Off = N_Lbl_O
    nSMR3   = SMR3set.shape[0]

    NodSet_y = np.zeros((0, 3))
    Elems3_y = np.zeros((0, 9))
    Elems4_y = np.zeros((0, 11))
    Bound_y  = np.zeros((0, 2))

    for iSMR3 in range(nSMR3):
        SMR3t = SMR3set[iSMR3, :]
        _, NodSet, Elems3, Elems4, Bound = gen_SMR3(
            KeyPts, SMR3t[0:6], DscrInsts, SMR3t[6:8],
            N_Label_Off, SMR3t[8], SMR3t[9], SMR3t[10]
        )
        N_Label_Off = int(NodSet[:, 0].max())
        NodSet_y = np.vstack([NodSet_y, NodSet])
        Elems3_y = np.vstack([Elems3_y, Elems3])
        Elems4_y = np.vstack([Elems4_y, Elems4])
        Bound_y  = np.vstack([Bound_y,  Bound ])

    max_x = np.abs(NodSet_y[:, 1]).max()
    max_y = np.abs(NodSet_y[:, 2]).max()
    tol   = 1e-12 * np.sqrt(max_x**2 + max_y**2)
    cnods = find_cnodes(NodSet_y, tol)

    # Both calls see the same Bound_y, so the second Bound_z is the result
    Elems3_x, Bound_z = kill_cnodes(Elems3_y, 6, Bound_y, cnods)
    Elems4_x, Bound_z = kill_cnodes(Elems4_y, 8, Bound_y, cnods)

    topo_nods3        = find_tnodes3(NodSet_y, Elems3_x)
    topo_nods4        = find_tnodes4(NodSet_y, Elems4_x)
    topo_nodes        = topo_nods3 + topo_nods4
    dead_node_indices = np.where(topo_nodes == 0)[0]
    NodSet_x          = np.delete(NodSet_y, dead_node_indices, axis=0)

    Bound_x = cncl_bsegs(Bound_z)

    print(f"\n Function gen_SMR3_set completed:\n")
    print(f" Total nodes generated:              {NodSet_y.shape[0]:4d}")
    print(f" Coincident node pairs found:        {cnods.shape[0]:4d}")
    print(f" Final nodes retained:               {NodSet_x.shape[0]:4d}")
    print(f" Final boundary segments:            {Bound_x.shape[0]:4d}")

    return NodSet_x, Elems3_x, Elems4_x, Bound_x


# Clean-up and areas

def tidy_up(NodSet_0, Elems3_0, Elems4_0, Bound_0):
    """
    Final clean-up of a labelled model: merge coincident nodes, drop unused
    nodes and remove internal boundary segments.

    Returns
    -------
    NodSet_1, Elems3_1, Elems4_1, Bound_1
    """
    max_x = np.abs(NodSet_0[:, 1]).max()
    max_y = np.abs(NodSet_0[:, 2]).max()
    tol   = 1e-12 * np.sqrt(max_x**2 + max_y**2)
    cnods = find_cnodes(NodSet_0, tol)

    Elems3_1, Bound_2 = kill_cnodes(Elems3_0, 6, Bound_0, cnods)
    Elems4_1, Bound_2 = kill_cnodes(Elems4_0, 8, Bound_0, cnods)

    topo_nods3        = find_tnodes3(NodSet_0, Elems3_1)
    topo_nods4        = find_tnodes4(NodSet_0, Elems4_1)
    topo_nodes        = topo_nods3 + topo_nods4
    dead_node_indices = np.where(topo_nodes == 0)[0]
    NodSet_1          = np.delete(NodSet_0, dead_node_indices, axis=0)

    Bound_1 = cncl_bsegs(Bound_2)

    print(f"\n Function tidy_up completed:\n")
    print(f" Input nodes:                        {NodSet_0.shape[0]:4d}")
    print(f" Coincident node pairs found:        {cnods.shape[0]:4d}")
    print(f" Final nodes retained:               {NodSet_1.shape[0]:4d}")
    print(f" Final boundary segments:            {Bound_1.shape[0]:4d}")

    return NodSet_1, Elems3_1, Elems4_1, Bound_1


def model_area(NodSet, Elems3, Elems4, MatLabl):
    """Total area of the elements with material label MatLabl (labelled model)."""
    m_area = 0.0

    for iele3 in range(Elems3.shape[0]):
        ele3 = Elems3[iele3, :]
        if ele3[7] == MatLabl:
            Indxs      = findlabl(NodSet, ele3[0:6]).ravel()
            area_t, _  = ele6_area(NodSet[Indxs, 1], NodSet[Indxs, 2])
            m_area    += area_t

    for iele4 in range(Elems4.shape[0]):
        ele4 = Elems4[iele4, :]
        if ele4[9] == MatLabl:
            Indxs      = findlabl(NodSet, ele4[0:8]).ravel()
            area_t, _  = ele8_area(NodSet[Indxs, 1], NodSet[Indxs, 2])
            m_area    += area_t

    return m_area


def print_nodset(NodSet):
    """Print a NodeSet as  label : (x, y)."""
    for inod in range(NodSet.shape[0]):
        print(f' {int(NodSet[inod, 0]):6d}  :  ( {NodSet[inod, 1]:9.6f} , {NodSet[inod, 2]:9.6f} )')