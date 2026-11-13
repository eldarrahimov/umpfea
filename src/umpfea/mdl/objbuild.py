"""
Assembly and static condensation of one machine part (stator or rotor).
"""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from umpfea.geom.labls import findlabl
from umpfea.mdl.build_k import ele6_calx, ele8_calx


def _find_row(table, labl, what, iele):
    indx = int(findlabl(table, [[labl]])[0, 0])
    if indx < 0:
        raise ValueError(f'objbuild: unknown {what} label {labl:g} in element {iele}')
    return indx


def objbuild(obj_model, name):
    """
    Assemble the FE model of one part and condense out the interior DOFs.

    Parameters
    ----------
    obj_model : ObjectModel
        Dereferenced model (0-based node indices), with
          NodeSet : (NDoF, 2)     [x, y] (m)
          Elems3  : (NEle3, 9)    [n0..n5, PD, Mat, CMap]
          Elems4  : (NEle4, 11)   [n0..n7, PD, Mat, CMap]
          PDirs   : (NPD, 4)      [label, flag, u, v]
          MProps  : (NMat, >=5)   [label, mu1, mu2, rem1, rem2, ...]
          CMaps   : (NCM, 1+NCC)  [label, c1..cNCC]
          MstrNds : (NMstr,)      airgap-facing master nodes
          CnstCmm : (NCnCm,)      nodes to be commoned later
    name : str
        Used in progress messages.

    Returns
    -------
    K_mat   : (NMs, NMs) ndarray
        Reduced stiffness. DOFs are MstrNds followed by CnstCmm.
    v_PM    : (NMs,) ndarray
        PM excitation.
    T_mat   : (NMs, NCC) ndarray
        Circuit-current coupling.
    SlavNds : (NSl,) ndarray
        Indices of the condensed-out nodes.
    v_PMx   : (NSl,) ndarray
        Slave PM term, for back-substitution.
    T_matx  : (NSl, NCC) ndarray
        Slave current term, for back-substitution.
    T_red   : (NDoF, NMs) ndarray
        Reduction matrix (full DOFs = T_red @ master DOFs).

    All flux quantities are per unit depth (Wb/m).
    """
    t_start = time.perf_counter()

    NodeSet = obj_model.NodeSet
    Elems3  = obj_model.Elems3
    Elems4  = obj_model.Elems4
    PDirs   = obj_model.PDirs
    MProps  = obj_model.MProps
    CMaps   = obj_model.CMaps
    MstrNds = obj_model.MstrNds.astype(int).ravel()
    CnstCmm = obj_model.CnstCmm.astype(int).ravel()

    NDoF  = NodeSet.shape[0]
    NEle4 = Elems4.shape[0]
    NEle3 = Elems3.shape[0]
    NMstr = MstrNds.shape[0]
    NCnCm = CnstCmm.shape[0]
    NCC   = CMaps.shape[1] - 1   # number of independent circuit currents

    print(f'\nobjbuild :: addressing <{name}>')
    print(f'            number of elem3s to process   = {NEle3:5d}')
    print(f'            number of elem4s to process   = {NEle4:5d}')
    print(f'            number of nodes in the model  = {NDoF:5d}')
    print(f'            number of master DoFs         = {NMstr:5d}')
    print(f'            number of commoned DoFs       = {NCnCm:5d}\n')

    # COO triplets for the sparse stiffness matrix
    k_rows = []
    k_cols = []
    k_vals = []
    v_PM1  = np.zeros(NDoF)
    T_mat1 = np.zeros((NDoF, NCC))

    # Quad elements
    t0 = time.perf_counter()
    for iele in range(NEle4):
        nod_inds = Elems4[iele, :8].astype(int)
        XNds = NodeSet[nod_inds, 0].reshape(8, 1)
        YNds = NodeSet[nod_inds, 1].reshape(8, 1)

        PD_labl  = Elems4[iele, 8]
        PD_indx  = _find_row(PDirs, PD_labl, 'PDir', iele)
        PD_Data  = PDirs[PD_indx, 1:4].reshape(1, 3)

        Mat_labl = Elems4[iele, 9]
        Mat_indx = _find_row(MProps, Mat_labl, 'material', iele)
        Mat_Data = MProps[Mat_indx, 1:5].reshape(1, 4)

        # Negative CMap label reverses the current
        CMp_labl = Elems4[iele, 10]
        if abs(CMp_labl) == 0:
            CMp_Data = np.zeros((1, NCC))
        else:
            CMp_indx = _find_row(CMaps, abs(CMp_labl), 'CMap', iele)
            CMp_Data = CMaps[CMp_indx, 1:NCC+1].reshape(1, NCC) * np.sign(CMp_labl)

        k_e, v_e, T_e = ele8_calx(XNds, YNds, PD_Data, Mat_Data, CMp_Data)

        ii, jj = np.meshgrid(nod_inds, nod_inds, indexing='ij')
        k_rows.append(ii.ravel())
        k_cols.append(jj.ravel())
        k_vals.append(k_e.ravel())
        v_PM1[nod_inds]     += v_e.ravel()
        T_mat1[nod_inds, :] += T_e   # (8, NCC)

        if (iele + 1) % 1000 == 0:
            print(f' Processed Elem4 #{iele+1:5d} of {NEle4:5d}')

    timtak4 = time.perf_counter() - t0
    print(f' objbuild :: <{name}> .. merged quad elems in {timtak4:.2f}s')

    # Triangle elements
    t0 = time.perf_counter()
    for iele in range(NEle3):
        nod_inds = Elems3[iele, :6].astype(int)
        XNds = NodeSet[nod_inds, 0].reshape(6, 1)
        YNds = NodeSet[nod_inds, 1].reshape(6, 1)

        PD_labl  = Elems3[iele, 6]
        PD_indx  = _find_row(PDirs, PD_labl, 'PDir', iele)
        PD_Data  = PDirs[PD_indx, 1:4].reshape(1, 3)

        Mat_labl = Elems3[iele, 7]
        Mat_indx = _find_row(MProps, Mat_labl, 'material', iele)
        Mat_Data = MProps[Mat_indx, 1:5].reshape(1, 4)

        CMp_labl = Elems3[iele, 8]
        if abs(CMp_labl) == 0:
            CMp_Data = np.zeros((1, NCC))
        else:
            CMp_indx = _find_row(CMaps, abs(CMp_labl), 'CMap', iele)
            CMp_Data = CMaps[CMp_indx, 1:NCC+1].reshape(1, NCC) * np.sign(CMp_labl)

        k_e, v_e, T_e = ele6_calx(XNds, YNds, PD_Data, Mat_Data, CMp_Data)

        ii, jj = np.meshgrid(nod_inds, nod_inds, indexing='ij')
        k_rows.append(ii.ravel())
        k_cols.append(jj.ravel())
        k_vals.append(k_e.ravel())
        v_PM1[nod_inds]     += v_e.ravel()
        T_mat1[nod_inds, :] += T_e   # (6, NCC)

        if (iele + 1) % 1000 == 0:
            print(f' Processed Elem3 #{iele+1:5d} of {NEle3:5d}')

    timtak3 = time.perf_counter() - t0
    print(f' objbuild :: <{name}> .. merged triang elems in {timtak3:.2f}s')

    # Duplicate COO entries are summed
    K_mat1 = sp.csr_matrix(
        (np.concatenate(k_vals),
         (np.concatenate(k_rows), np.concatenate(k_cols))),
        shape=(NDoF, NDoF)
    )

    # Masters = MstrNds + CnstCmm, everything else is a slave
    print(f' objbuild :: <{name}> .. now partitioning and reducing')
    t0 = time.perf_counter()

    AllMstr = np.concatenate([MstrNds, CnstCmm])  # master + common
    SlavNds = np.setdiff1d(np.arange(NDoF), AllMstr)
    NMs = len(AllMstr)
    NSl = len(SlavNds)

    print(f' objbuild :: <{name}> .. Number of DoFs in total = {NDoF:5d}')
    print(f' objbuild :: <{name}> .. Number of master DoFs   = {NMs:5d}')
    print(f' objbuild :: <{name}> .. Number of slave DoFs    = {NSl:5d}')

    K_SS = K_mat1[np.ix_(SlavNds, SlavNds)]       # sparse (NSl, NSl)
    K_SM = K_mat1[np.ix_(SlavNds, AllMstr)]        # sparse (NSl, NMs)

    # Static condensation: T_slav = -K_SS^-1 K_SM
    lu = spla.splu(K_SS.tocsc())
    T_slav = lu.solve(-K_SM.toarray())   # (NSl, NMs)

    T_red = np.zeros((NDoF, NMs))
    T_red[AllMstr, :] = np.eye(NMs)
    T_red[SlavNds, :] = T_slav

    timtakr = time.perf_counter() - t0
    print(f' Time taken to form 1st reduction matrix = {timtakr:.2f}(s)')

    print(f' objbuild :: <{name}> .. now preparing all outputs')
    t0 = time.perf_counter()

    KT    = K_mat1 @ T_red          # (NDoF, NMs)
    K_mat = T_red.T @ KT            # (NMs, NMs)
    v_PM  = T_red.T @ v_PM1         # (NMs,)
    T_mat = T_red.T @ T_mat1        # (NMs, NCC)

    # Slave-side terms for back-substitution
    v_PMx  = lu.solve(-v_PM1[SlavNds])          # (NSl,)
    T_matx = lu.solve(-T_mat1[SlavNds, :])      # (NSl, NCC)

    timtake = time.perf_counter() - t0
    print(f' objbuild :: <{name}> .. Time to form all outputs = {timtake:.2f}(s)')

    total_time = time.perf_counter() - t_start
    print(f' objbuild :: <{name}> .. total time absorbed = {total_time:.2f}(s)')

    return K_mat, v_PM, T_mat, SlavNds, v_PMx, T_matx, T_red
