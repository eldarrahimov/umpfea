import numpy as np
from umpfea.geom.labls import findlabl
from .smrs  import find_cnodes, kill_cnodes, find_tnodes3, find_tnodes4, cncl_bsegs


def bpc_chek(nbay, nndxs, iprp, oldv):
    """
    Check that a property label occurs a multiple of nbay times
    (used by bay_props_ch).
    """
    if nndxs % nbay != 0:
        raise ValueError(
            f' ERROR trapped in <bay_props_ch>:\n'
            f' Number of bays advised    = {nbay:3d}\n'
            f' Property index = {iprp:2d} and property value = {oldv:6d}\n'
            f' Number of instances found = {nndxs:6d}'
        )


def replic_nodset(NodeSet, angle_rad, label_offset, centre_xy):
    """
    Rotated copy of a node set with label_offset added to every label.

    Parameters
    ----------
    NodeSet      : (N, 3) ndarray  [label, x, y]
    angle_rad    : float, rotation angle (rad)
    label_offset : float
    centre_xy    : (2,) ndarray, rotation centre

    Returns
    -------
    NodeSet_new : (N, 3) ndarray
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R    = np.array([[c, -s], [s, c]])

    xy_rel = NodeSet[:, 1:3] - centre_xy
    xy_rot = (R @ xy_rel.T).T + centre_xy

    NodeSet_new          = NodeSet.copy()
    NodeSet_new[:, 0]   += label_offset
    NodeSet_new[:, 1:3]  = xy_rot

    return NodeSet_new


def multi_replic(NodSet_bay, Elems3_bay, Elems4_bay, Bound_bay,
                 KeyPts, KP_lbl1, KP_lbl2, KP_ctr_lbl, n_replic):
    """
    Replicate one bay n_replic times around the machine centre and merge
    the coincident nodes at the bay interfaces.

    The bay angle is the angle between KP_lbl1 and KP_lbl2 about KP_ctr_lbl.
    Total number of bays is n_replic + 1.

    Returns
    -------
    NodSet_out, Elems3_out, Elems4_out, Bound_out
    """
    idx1  = findlabl(KeyPts, np.atleast_2d(KP_lbl1)).ravel()[0]
    idx2  = findlabl(KeyPts, np.atleast_2d(KP_lbl2)).ravel()[0]
    idx_c = findlabl(KeyPts, np.atleast_2d(KP_ctr_lbl)).ravel()[0]

    centre_xy = KeyPts[idx_c, 1:3]
    xy1       = KeyPts[idx1, 1:3] - centre_xy
    xy2       = KeyPts[idx2, 1:3] - centre_xy
    bay_angle = np.arctan2(xy2[1], xy2[0]) - np.arctan2(xy1[1], xy1[0])

    # Per-bay label offset keeps labels unique
    max_lbl = int(NodSet_bay[:, 0].max())

    NodSet_all = NodSet_bay.copy()
    Elems3_all = Elems3_bay.copy()
    Elems4_all = Elems4_bay.copy()
    Bound_all  = Bound_bay.copy()

    for i in range(1, n_replic + 1):
        lbl_off = i * max_lbl
        angle_i = i * bay_angle

        NodSet_i        = replic_nodset(NodSet_bay, angle_i, lbl_off, centre_xy)
        Elems3_i        = Elems3_bay.copy()
        Elems3_i[:, :6] += lbl_off      # topology columns only
        Elems4_i        = Elems4_bay.copy()
        Elems4_i[:, :8] += lbl_off
        Bound_i         = Bound_bay.copy() + lbl_off

        NodSet_all = np.vstack([NodSet_all, NodSet_i])
        Elems3_all = np.vstack([Elems3_all, Elems3_i])
        Elems4_all = np.vstack([Elems4_all, Elems4_i])
        Bound_all  = np.vstack([Bound_all,  Bound_i])

    # Looser tolerance than within a bay, since the copies come from rotation
    max_x = np.abs(NodSet_all[:, 1]).max()
    max_y = np.abs(NodSet_all[:, 2]).max()
    tol   = 1e-10 * np.sqrt(max_x**2 + max_y**2)
    cnods = find_cnodes(NodSet_all, tol)

    Elems3_out, Bound_tmp = kill_cnodes(Elems3_all, 6, Bound_all,  cnods)
    Elems4_out, Bound_tmp = kill_cnodes(Elems4_all, 8, Bound_tmp,  cnods)

    # Drop nodes no longer used by any element
    topo_flags   = find_tnodes3(NodSet_all, Elems3_out) \
                 + find_tnodes4(NodSet_all, Elems4_out)
    dead_indices = np.where(topo_flags == 0)[0]
    NodSet_out   = np.delete(NodSet_all, dead_indices, axis=0)

    Bound_out = cncl_bsegs(Bound_tmp)

    print(f'\n Function multi_replic completed:')
    print(f' Bays created:                    {n_replic + 1:4d}')
    print(f' Coincident node pairs merged:    {cnods.shape[0]:4d}')
    print(f' Final nodes:                     {NodSet_out.shape[0]:4d}')
    print(f' Final boundary segments:         {Bound_out.shape[0]:4d}')

    return NodSet_out, Elems3_out, Elems4_out, Bound_out


def bay_props_ch(Elems3, Elems4, iprp, srchv, incr, nbay):
    """
    Renumber an element property label bay by bay.

    Elements carrying a label in srchv are assigned to bays in the order
    they appear (bays stacked as by multi_replic).

    Parameters
    ----------
    Elems3, Elems4 : element arrays
    iprp  : 1 = PDir, 2 = material, 3 = CMap
    srchv : label value(s) to change, e.g. [5701, 5702]
    incr  : incr > 0: new = (|old| + (bay-1)*incr) * sign(old)
            incr < 0: sign alternates from bay to bay
    nbay  : total number of bays

    Returns
    -------
    Elems3_out, Elems4_out : updated copies
    n3, n4 : number of matches in Elems3 / Elems4
    """
    col_map = {1: (6, 8), 2: (7, 9), 3: (8, 10)}
    col_i3, col_i4 = col_map[iprp]

    srchv      = np.asarray(srchv, dtype=float)
    Elems3_out = Elems3.copy()
    Elems4_out = Elems4.copy()

    def _apply(Elems_out, col):
        mask = np.zeros(Elems_out.shape[0], dtype=bool)
        for v in srchv:
            mask |= (Elems_out[:, col] == v)
        indxs = np.where(mask)[0]
        n     = len(indxs)
        if n == 0:
            return n

        bpc_chek(nbay, n, iprp, int(srchv[0]))

        values   = Elems_out[indxs, col]
        seq      = np.arange(1, n + 1)
        baynums  = np.ceil((nbay * seq - 0.5) / n).astype(int)

        if incr < 0:
            Elems_out[indxs, col] = values * (np.mod(baynums, 2) * 2 - 1)
        else:
            Elems_out[indxs, col] = (np.abs(values) + (baynums - 1) * incr) \
                                    * np.sign(values)
        return n

    n3 = _apply(Elems3_out, col_i3)
    n4 = _apply(Elems4_out, col_i4)

    return Elems3_out, Elems4_out, n3, n4


def node_subset(NodeSet, KeyPts, KP_radius_lbl, angle_param, KP_ctr_lbl, tol):
    """
    Nodes lying at the radius of KeyPoint KP_radius_lbl about KP_ctr_lbl.

    Parameters
    ----------
    NodeSet       : (N, 3) ndarray  [label, x, y]
    KeyPts        : KeyPoint table [label, x, y, ...]
    KP_radius_lbl : KeyPoint that defines the radius
    angle_param   : unused, pass 0
    KP_ctr_lbl    : centre KeyPoint
    tol           : radial tolerance (m)

    Returns
    -------
    indxs  : int ndarray, rows in NodeSet
    labels : ndarray, labels of those nodes
    """
    idx_r     = findlabl(KeyPts, np.atleast_2d(KP_radius_lbl)).ravel()[0]
    idx_c     = findlabl(KeyPts, np.atleast_2d(KP_ctr_lbl)).ravel()[0]
    centre_xy = KeyPts[idx_c, 1:3]
    kp_r_xy   = KeyPts[idx_r, 1:3]
    target_r  = np.sqrt(np.sum((kp_r_xy - centre_xy) ** 2))

    dxy    = NodeSet[:, 1:3] - centre_xy
    radii  = np.sqrt(dxy[:, 0] ** 2 + dxy[:, 1] ** 2)
    indxs  = np.where(np.abs(radii - target_r) < tol)[0]
    labels = NodeSet[indxs, 0]

    return indxs, labels
