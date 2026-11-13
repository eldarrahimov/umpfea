"""
Element functions for 6-node triangles and 8-node serendipity quads.

  ele6_calx, ele8_calx      stiffness k (n x n), PM excitation v_PM (n x 1)
                            and current coupling T (n x NCC), per unit
                            depth (Wb/m)
  eval_6_sfs, eval_8_sfs    shape functions
  ele6_area, ele8_area      element area and shape function integrals
  ele6AG_calx, ele8AG_calx  free-space stiffness (airgap elements)
"""

import numpy as np


def ele6_calx(Node_X, Node_Y, PD_Data, Mat_Data, CMp_Data):
    """
    6-node triangle element matrices.

    Parameters
    ----------
    Node_X, Node_Y : (6, 1) ndarray
        Nodal coordinates (m).
    PD_Data : (1, 3) ndarray
        [flag, u, v]. flag 0: (u, v) is in reference (ksi, eta) space;
        flag 1: (u, v) is in global (x, y).
    Mat_Data : (1, 4) ndarray
        [mu_r1, mu_r2, Br1, Br2] along / across the principal direction.
    CMp_Data : (1, NCC) ndarray
        Current density per unit circuit current.

    Returns
    -------
    k_ele : (6, 6) ndarray
    v_PM  : (6, 1) ndarray
    T     : (6, NCC) ndarray
    """
    # Reluctivity in the principal-direction frame
    Emat0 = np.diag([1.0 / Mat_Data[0, 0],
                     1.0 / Mat_Data[0, 1]]) * (1.0 / (4e-7 * np.pi))

    v_rem_flx0 = np.array([[Mat_Data[0, 2]],
                            [Mat_Data[0, 3]]])

    # Rotates a gradient by 90 deg (grad A -> B)
    S = np.array([[0.0, -1.0],
                  [1.0,  0.0]])

    k_ele = np.zeros((6, 6))
    v_PM  = np.zeros((6, 1))
    T     = np.zeros((6, 1))   # becomes (6, NCC) on the first add

    # 7-point Gauss rule; weights halved because the reference triangle has area 0.5
    coors = np.array([0.1012865073235,
                      0.7974269853531,
                      0.4701420641051,
                      0.0597158717898,
                      0.3333333333333])
    coefs = np.array([0.1259391805448,
                      0.1323941527885,
                      0.2250000000000]) / 2.0

    GSSP = np.array([[coors[0], coors[0], coefs[0]],
                     [coors[1], coors[0], coefs[0]],
                     [coors[0], coors[1], coefs[0]],
                     [coors[2], coors[3], coefs[1]],
                     [coors[2], coors[2], coefs[1]],
                     [coors[3], coors[2], coefs[1]],
                     [coors[4], coors[4], coefs[2]]])

    for igp in range(7):
        ksi = GSSP[igp, 0]
        eta = GSSP[igp, 1]

        tw_6 = np.array([[2.0 * (ksi + eta - 1.0) * (ksi + eta - 0.5)],
                          [1.0 * eta * (2.0 * eta - 1.0)],
                          [1.0 * ksi * (2.0 * ksi - 1.0)],
                          [-4.0 * eta * (ksi + eta - 1.0)],
                          [4.0 * ksi * eta],
                          [-4.0 * ksi * (ksi + eta - 1.0)]])

        tw_6_diff_ksi = np.array([[4.0 * ksi + 4.0 * eta - 3.0],
                                   [0.0],
                                   [4.0 * ksi - 1.0],
                                   [-4.0 * eta],
                                   [4.0 * eta],
                                   [-8.0 * ksi - 4.0 * eta + 4.0]])

        tw_6_diff_eta = np.array([[4.0 * ksi + 4.0 * eta - 3.0],
                                   [4.0 * eta - 1.0],
                                   [0.0],
                                   [-4.0 * ksi - 8.0 * eta + 4.0],
                                   [4.0 * ksi],
                                   [-4.0 * ksi]])

        Bv     = np.vstack((tw_6_diff_ksi.T, tw_6_diff_eta.T))   # (2,6)
        Jaco   = Bv @ np.hstack((Node_X, Node_Y))                 # (2,2)
        DetJ   = np.linalg.det(Jaco)
        JaciBv = np.linalg.solve(Jaco, Bv)                        # (2,6)
        factor = DetJ * GSSP[igp, 2]

        # Principal direction in (x, y)
        v_pd = np.array([PD_Data[0, 1], PD_Data[0, 2]])
        if PD_Data[0, 0] == 0:
            v_pd_xy = Jaco.T @ v_pd
        else:
            v_pd_xy = v_pd
        v_pd_xy = v_pd_xy / np.sqrt(v_pd_xy @ v_pd_xy)

        R2x2 = np.array([[ v_pd_xy[0], -v_pd_xy[1]],
                          [ v_pd_xy[1],  v_pd_xy[0]]])
        Emat    = R2x2 @ Emat0 @ R2x2.T
        SJaciBv = S @ JaciBv

        k_ele     = k_ele + factor * (SJaciBv.T @ Emat @ SJaciBv)
        v_rem_flx = R2x2 @ v_rem_flx0
        v_PM      = v_PM  + factor * (SJaciBv.T @ (Emat @ v_rem_flx))
        T         = T     + factor * tw_6 @ CMp_Data

    return k_ele, v_PM, T


def ele8_calx(Node_X, Node_Y, PD_Data, Mat_Data, CMp_Data):
    """
    8-node serendipity quad element matrices.

    Same arguments as ele6_calx, with (8, 1) node coordinates.

    Returns
    -------
    k_ele : (8, 8) ndarray
    v_PM  : (8, 1) ndarray
    T     : (8, NCC) ndarray
    """
    # Reluctivity in the principal-direction frame
    Emat0 = np.diag([1.0 / Mat_Data[0, 0],
                     1.0 / Mat_Data[0, 1]]) * (1.0 / (4e-7 * np.pi))

    v_rem_flx0 = np.array([[Mat_Data[0, 2]],
                            [Mat_Data[0, 3]]])

    S = np.array([[0.0, -1.0],
                  [1.0,  0.0]])

    k_ele = np.zeros((8, 8))
    v_PM  = np.zeros((8, 1))
    T     = np.zeros((8, 1))   # becomes (8, NCC) on the first add

    # 3x3 Gauss rule on [-1, 1]^2
    coors = np.array([0.774596669241483, 0.000000000000000])
    coefs = np.array([0.555555555555556, 0.888888888888889])

    GSSP = np.array([[-coors[0], -coors[0], coefs[0] * coefs[0]],
                     [ coors[1], -coors[0], coefs[1] * coefs[0]],
                     [ coors[0], -coors[0], coefs[0] * coefs[0]],
                     [-coors[0],  coors[1], coefs[0] * coefs[1]],
                     [ coors[1],  coors[1], coefs[1] * coefs[1]],
                     [ coors[0],  coors[1], coefs[0] * coefs[1]],
                     [-coors[0],  coors[0], coefs[0] * coefs[0]],
                     [ coors[1],  coors[0], coefs[1] * coefs[0]],
                     [ coors[0],  coors[0], coefs[0] * coefs[0]]])

    for igp in range(9):
        ksi = GSSP[igp, 0]
        eta = GSSP[igp, 1]

        tw_8 = np.array([[0.25 * (1 - ksi) * (1 - eta) * (-1 - ksi - eta)],
                          [0.25 * (1 - ksi) * (1 + eta) * (-1 - ksi + eta)],
                          [0.25 * (1 + ksi) * (1 + eta) * (-1 + ksi + eta)],
                          [0.25 * (1 + ksi) * (1 - eta) * (-1 + ksi - eta)],
                          [0.50 * (1 - ksi) * (1 - eta * eta)],
                          [0.50 * (1 + eta) * (1 - ksi * ksi)],
                          [0.50 * (1 + ksi) * (1 - eta * eta)],
                          [0.50 * (1 - eta) * (1 - ksi * ksi)]])

        tw_8_diff_ksi = np.array([[0.25 * (2*ksi + eta - 2*ksi*eta - eta*eta)],
                                   [0.25 * (2*ksi - eta + 2*ksi*eta - eta*eta)],
                                   [0.25 * (2*ksi + eta + 2*ksi*eta + eta*eta)],
                                   [0.25 * (2*ksi - eta - 2*ksi*eta + eta*eta)],
                                   [0.50 * (-1 + eta*eta)],
                                   [0.50 * (-2*ksi - 2*ksi*eta)],
                                   [0.50 * ( 1 - eta*eta)],
                                   [0.50 * (-2*ksi + 2*ksi*eta)]])

        tw_8_diff_eta = np.array([[0.25 * (2*eta + ksi - 2*ksi*eta - ksi*ksi)],
                                   [0.25 * (2*eta - ksi - 2*ksi*eta + ksi*ksi)],
                                   [0.25 * (2*eta + ksi + 2*ksi*eta + ksi*ksi)],
                                   [0.25 * (2*eta - ksi + 2*ksi*eta - ksi*ksi)],
                                   [0.50 * (-2*eta + 2*ksi*eta)],
                                   [0.50 * ( 1 - ksi*ksi)],
                                   [0.50 * (-2*eta - 2*ksi*eta)],
                                   [0.50 * (-1 + ksi*ksi)]])

        Bv     = np.vstack((tw_8_diff_ksi.T, tw_8_diff_eta.T))   # (2,8)
        Jaco   = Bv @ np.hstack((Node_X, Node_Y))                 # (2,2)
        DetJ   = np.linalg.det(Jaco)
        JaciBv = np.linalg.solve(Jaco, Bv)                        # (2,8)
        factor = DetJ * GSSP[igp, 2]

        # Principal direction in (x, y)
        v_pd = np.array([PD_Data[0, 1], PD_Data[0, 2]])
        if PD_Data[0, 0] == 0:
            v_pd_xy = Jaco.T @ v_pd
        else:
            v_pd_xy = v_pd
        v_pd_xy = v_pd_xy / np.sqrt(v_pd_xy @ v_pd_xy)

        R2x2 = np.array([[ v_pd_xy[0], -v_pd_xy[1]],
                          [ v_pd_xy[1],  v_pd_xy[0]]])
        Emat    = R2x2 @ Emat0 @ R2x2.T
        SJaciBv = S @ JaciBv

        k_ele     = k_ele + factor * (SJaciBv.T @ Emat @ SJaciBv)
        v_rem_flx = R2x2 @ v_rem_flx0
        v_PM      = v_PM  + factor * (SJaciBv.T @ (Emat @ v_rem_flx))
        T         = T     + factor * tw_8 @ CMp_Data

    return k_ele, v_PM, T


def eval_6_sfs(ksi, eta, indx=0):
    """
    6-node triangle shape functions at (ksi, eta), on the (-1, 1) reference
    triangle used by gen_SMR3.

    Node ordering:
        2
        4  5
        1  6  3

    indx:
        0   : all 6 values at a scalar point, shape (6,)
        -1  : all 6 for array input, shape (..., 6)
        1-6 : that single function
    """
    if indx == 0:
        k = float(ksi)
        e = float(eta)
        f1 = (k + e) * (1.0 + (k + e)) / 2.0
        f2 = (1.0 + e) * e / 2.0
        f3 = (1.0 + k) * k / 2.0
        f4 = (1.0 + e) * (-(k + e))
        f5 = (1.0 + e) * (1.0 + k)
        f6 = (1.0 + k) * (-(k + e))
        return np.array([f1, f2, f3, f4, f5, f6])

    ksi = np.asarray(ksi, dtype=float)
    eta = np.asarray(eta, dtype=float)

    if indx == -1:
        f1 = (ksi + eta) * (1.0 + (ksi + eta)) / 2.0
        f2 = (1.0 + eta) * eta / 2.0
        f3 = (1.0 + ksi) * ksi / 2.0
        f4 = (1.0 + eta) * (-(ksi + eta))
        f5 = (1.0 + eta) * (1.0 + ksi)
        f6 = (1.0 + ksi) * (-(ksi + eta))
        return np.stack([f1, f2, f3, f4, f5, f6], axis=-1)

    if indx == 1:
        return (ksi + eta) * (1.0 + (ksi + eta)) / 2.0
    elif indx == 2:
        return (1.0 + eta) * eta / 2.0
    elif indx == 3:
        return (1.0 + ksi) * ksi / 2.0
    elif indx == 4:
        return (1.0 + eta) * (-(ksi + eta))
    elif indx == 5:
        return (1.0 + eta) * (1.0 + ksi)
    elif indx == 6:
        return (1.0 + ksi) * (-(ksi + eta))
    else:
        raise ValueError(f"Invalid indx={indx} in eval_6_sfs; must be -1, 0, or 1–6.")


def eval_8_sfs(ksi, eta, indx=0):
    """
    8-node serendipity quad shape functions at (ksi, eta).

    Node ordering:
        2  6  3
        5     7
        1  8  4

    indx:
        0   : all 8 values at a scalar point, shape (8,)
        -1  : all 8 for array input, shape (..., 8)
        1-8 : that single function
    """
    if indx == 0:
        k = float(ksi)
        e = float(eta)
        f5 = (1.0 - e) * (1.0 + e) * (1.0 - k) / 2.0
        f6 = (1.0 - k) * (1.0 + k) * (1.0 + e) / 2.0
        f7 = (1.0 - e) * (1.0 + e) * (1.0 + k) / 2.0
        f8 = (1.0 - k) * (1.0 + k) * (1.0 - e) / 2.0
        f1 = (1.0 - k) * (1.0 - e) / 4.0 - 0.5 * (f5 + f8)
        f2 = (1.0 - k) * (1.0 + e) / 4.0 - 0.5 * (f5 + f6)
        f3 = (1.0 + k) * (1.0 + e) / 4.0 - 0.5 * (f6 + f7)
        f4 = (1.0 + k) * (1.0 - e) / 4.0 - 0.5 * (f7 + f8)
        return np.array([f1, f2, f3, f4, f5, f6, f7, f8])

    ksi = np.asarray(ksi, dtype=float)
    eta = np.asarray(eta, dtype=float)

    if indx == -1:
        f5 = (1.0 - eta) * (1.0 + eta) * (1.0 - ksi) / 2.0
        f6 = (1.0 - ksi) * (1.0 + ksi) * (1.0 + eta) / 2.0
        f7 = (1.0 - eta) * (1.0 + eta) * (1.0 + ksi) / 2.0
        f8 = (1.0 - ksi) * (1.0 + ksi) * (1.0 - eta) / 2.0
        f1 = (1.0 - ksi) * (1.0 - eta) / 4.0 - 0.5 * (f5 + f8)
        f2 = (1.0 - ksi) * (1.0 + eta) / 4.0 - 0.5 * (f5 + f6)
        f3 = (1.0 + ksi) * (1.0 + eta) / 4.0 - 0.5 * (f6 + f7)
        f4 = (1.0 + ksi) * (1.0 - eta) / 4.0 - 0.5 * (f7 + f8)
        return np.stack([f1, f2, f3, f4, f5, f6, f7, f8], axis=-1)

    if indx == 1:
        f5 = (1.0 - eta) * (1.0 + eta) * (1.0 - ksi) / 2.0
        f8 = (1.0 - ksi) * (1.0 + ksi) * (1.0 - eta) / 2.0
        return (1.0 - ksi) * (1.0 - eta) / 4.0 - 0.5 * (f5 + f8)
    elif indx == 2:
        f5 = (1.0 - eta) * (1.0 + eta) * (1.0 - ksi) / 2.0
        f6 = (1.0 - ksi) * (1.0 + ksi) * (1.0 + eta) / 2.0
        return (1.0 - ksi) * (1.0 + eta) / 4.0 - 0.5 * (f5 + f6)
    elif indx == 3:
        f6 = (1.0 - ksi) * (1.0 + ksi) * (1.0 + eta) / 2.0
        f7 = (1.0 - eta) * (1.0 + eta) * (1.0 + ksi) / 2.0
        return (1.0 + ksi) * (1.0 + eta) / 4.0 - 0.5 * (f6 + f7)
    elif indx == 4:
        f7 = (1.0 - eta) * (1.0 + eta) * (1.0 + ksi) / 2.0
        f8 = (1.0 - ksi) * (1.0 + ksi) * (1.0 - eta) / 2.0
        return (1.0 + ksi) * (1.0 - eta) / 4.0 - 0.5 * (f7 + f8)
    elif indx == 5:
        return (1.0 - eta) * (1.0 + eta) * (1.0 - ksi) / 2.0
    elif indx == 6:
        return (1.0 - ksi) * (1.0 + ksi) * (1.0 + eta) / 2.0
    elif indx == 7:
        return (1.0 - eta) * (1.0 + eta) * (1.0 + ksi) / 2.0
    elif indx == 8:
        return (1.0 - ksi) * (1.0 + ksi) * (1.0 - eta) / 2.0
    else:
        raise ValueError(f"Invalid indx={indx} in eval_8_sfs; must be -1, 0, or 1–8.")


def ele6_area(Node_X, Node_Y):
    """
    Area of a 6-node triangle (7-point Gauss rule).

    Returns
    -------
    A_ele : float, element area
    V_ele : (6,) ndarray, integral of each shape function
    """
    Node_X = np.asarray(Node_X, dtype=float).ravel()
    Node_Y = np.asarray(Node_Y, dtype=float).ravel()
    V_ele  = np.zeros(6)

    coors = np.array([0.1012865073235, 0.7974269853531,
                      0.4701420641051, 0.0597158717898, 0.3333333333333])
    coefs = np.array([0.1259391805448, 0.1323941527885, 0.2250000000000]) / 2.0

    GSSP = np.array([
        [coors[0], coors[0], coefs[0]],
        [coors[1], coors[0], coefs[0]],
        [coors[0], coors[1], coefs[0]],
        [coors[2], coors[3], coefs[1]],
        [coors[2], coors[2], coefs[1]],
        [coors[3], coors[2], coefs[1]],
        [coors[4], coors[4], coefs[2]],
    ])

    for igp in range(7):
        ksi = GSSP[igp, 0]
        eta = GSSP[igp, 1]

        tw_6 = np.array([
             2.0*(ksi + eta - 1)*(ksi + eta - 0.5),
             1.0*eta*(2.0*eta - 1.0),
             1.0*ksi*(2.0*ksi - 1.0),
            -4.0*eta*(ksi + eta - 1.0),
             4.0*ksi*eta,
            -4.0*ksi*(ksi + eta - 1.0),
        ])
        tw_6_dk = np.array([
             4.0*ksi + 4.0*eta - 3.0,
             0.0,
             4.0*ksi - 1.0,
            -4.0*eta,
             4.0*eta,
            -8.0*ksi - 4.0*eta + 4.0,
        ])
        tw_6_de = np.array([
             4.0*ksi + 4.0*eta - 3.0,
             4.0*eta - 1.0,
             0.0,
            -4.0*ksi - 8.0*eta + 4.0,
             4.0*ksi,
            -4.0*ksi,
        ])

        Bv   = np.vstack([tw_6_dk, tw_6_de])           # (2, 6)
        Jaco = Bv @ np.column_stack([Node_X, Node_Y])  # (2, 2)
        DetJ = np.linalg.det(Jaco)
        V_ele += DetJ * GSSP[igp, 2] * tw_6

    return float(np.sum(V_ele)), V_ele


def ele8_area(Node_X, Node_Y):
    """
    Area of an 8-node quad (3x3 Gauss rule).

    Returns
    -------
    A_ele : float, element area
    V_ele : (8,) ndarray, integral of each shape function
    """
    Node_X = np.asarray(Node_X, dtype=float).ravel()
    Node_Y = np.asarray(Node_Y, dtype=float).ravel()
    V_ele  = np.zeros(8)

    coors = np.array([0.774596669241483, 0.000000000000000])
    coefs = np.array([0.555555555555556, 0.888888888888889])

    GSSP = np.array([
        [-coors[0], -coors[0], coefs[0]*coefs[0]],
        [ coors[1], -coors[0], coefs[1]*coefs[0]],
        [ coors[0], -coors[0], coefs[0]*coefs[0]],
        [-coors[0],  coors[1], coefs[0]*coefs[1]],
        [ coors[1],  coors[1], coefs[1]*coefs[1]],
        [ coors[0],  coors[1], coefs[0]*coefs[1]],
        [-coors[0],  coors[0], coefs[0]*coefs[0]],
        [ coors[1],  coors[0], coefs[1]*coefs[0]],
        [ coors[0],  coors[0], coefs[0]*coefs[0]],
    ])

    for igp in range(9):
        ksi = GSSP[igp, 0]
        eta = GSSP[igp, 1]

        tw_8 = np.array([
            0.25*(1-ksi)*(1-eta)*(-1-ksi-eta),
            0.25*(1-ksi)*(1+eta)*(-1-ksi+eta),
            0.25*(1+ksi)*(1+eta)*(-1+ksi+eta),
            0.25*(1+ksi)*(1-eta)*(-1+ksi-eta),
            0.50*(1-ksi)*(1-eta**2),
            0.50*(1+eta)*(1-ksi**2),
            0.50*(1+ksi)*(1-eta**2),
            0.50*(1-eta)*(1-ksi**2),
        ])
        tw_8_dk = np.array([
            0.25*(2*ksi + eta - 2*ksi*eta - eta**2),
            0.25*(2*ksi - eta + 2*ksi*eta - eta**2),
            0.25*(2*ksi + eta + 2*ksi*eta + eta**2),
            0.25*(2*ksi - eta - 2*ksi*eta + eta**2),
            0.50*(-1 + eta**2),
            0.50*(-2*ksi - 2*ksi*eta),
            0.50*( 1 - eta**2),
            0.50*(-2*ksi + 2*ksi*eta),
        ])
        tw_8_de = np.array([
            0.25*(2*eta + ksi - 2*ksi*eta - ksi**2),
            0.25*(2*eta - ksi - 2*ksi*eta + ksi**2),
            0.25*(2*eta + ksi + 2*ksi*eta + ksi**2),
            0.25*(2*eta - ksi + 2*ksi*eta - ksi**2),
            0.50*(-2*eta + 2*ksi*eta),
            0.50*( 1 - ksi**2),
            0.50*(-2*eta - 2*ksi*eta),
            0.50*(-1 + ksi**2),
        ])

        Bv   = np.vstack([tw_8_dk, tw_8_de])           # (2, 8)
        Jaco = Bv @ np.column_stack([Node_X, Node_Y])  # (2, 2)
        DetJ = np.linalg.det(Jaco)
        V_ele += DetJ * GSSP[igp, 2] * tw_8

    return float(np.sum(V_ele)), V_ele


def ele6AG_calx(Node_X, Node_Y):
    """
    6-node triangle stiffness for free space (airgap elements).

    Reference node positions (ksi, eta):
      0 (0, 0), 1 (0, 1), 2 (1, 0), 3 (0, 0.5), 4 (0.5, 0.5), 5 (0.5, 0)

    Parameters
    ----------
    Node_X, Node_Y : (6,) ndarray
        Nodal coordinates (m).

    Returns
    -------
    k_ele : (6, 6) ndarray
    """
    Emat  = np.eye(2) * (1.0 / (4e-7 * np.pi))
    S     = np.array([[0.0, -1.0], [1.0, 0.0]])
    k_ele = np.zeros((6, 6))

    coors = np.array([0.1012865073235,
                      0.7974269853531,
                      0.4701420641051,
                      0.0597158717898,
                      0.3333333333333])
    coefs = np.array([0.1259391805448,
                      0.1323941527885,
                      0.2250000000000]) / 2   # reference triangle area is 0.5

    GSSP = np.array([
        [coors[0], coors[0], coefs[0]],
        [coors[1], coors[0], coefs[0]],
        [coors[0], coors[1], coefs[0]],
        [coors[2], coors[3], coefs[1]],
        [coors[2], coors[2], coefs[1]],
        [coors[3], coors[2], coefs[1]],
        [coors[4], coors[4], coefs[2]],
    ])

    for igp in range(7):
        ksi = GSSP[igp, 0]
        eta = GSSP[igp, 1]

        tw_6_diff_ksi = np.array([
             4.0*ksi + 4.0*eta - 3.0,
             0.0,
             4.0*ksi - 1.0,
            -4.0*eta,
             4.0*eta,
            -8.0*ksi - 4.0*eta + 4.0])

        tw_6_diff_eta = np.array([
             4.0*ksi + 4.0*eta - 3.0,
             4.0*eta - 1.0,
             0.0,
            -4.0*ksi - 8.0*eta + 4.0,
             4.0*ksi,
            -4.0*ksi])

        Bv     = np.vstack([tw_6_diff_ksi, tw_6_diff_eta])   # (2, 6)
        Jaco   = Bv @ np.column_stack([Node_X, Node_Y])       # (2, 2)
        DetJ   = np.linalg.det(Jaco)
        JaciBv = np.linalg.solve(Jaco, Bv)                    # (2, 6)
        factor = DetJ * GSSP[igp, 2]

        SJaciBv = S @ JaciBv
        k_ele  += factor * (SJaciBv.T @ Emat @ SJaciBv)

    return k_ele


def ele8AG_calx(Node_X, Node_Y):
    """
    8-node quad stiffness for free space (airgap elements).

    Parameters
    ----------
    Node_X, Node_Y : (8,) ndarray
        Nodal coordinates (m).

    Returns
    -------
    k_ele : (8, 8) ndarray
    """
    Emat = np.eye(2) * (1.0 / (4e-7 * np.pi))
    S    = np.array([[0.0, -1.0], [1.0, 0.0]])
    k_ele = np.zeros((8, 8))

    coors = np.array([0.774596669241483, 0.0])
    coefs = np.array([0.555555555555556, 0.888888888888889])

    GSSP = np.array([[-coors[0], -coors[0], coefs[0]*coefs[0]],
                     [ coors[1], -coors[0], coefs[1]*coefs[0]],
                     [ coors[0], -coors[0], coefs[0]*coefs[0]],
                     [-coors[0],  coors[1], coefs[0]*coefs[1]],
                     [ coors[1],  coors[1], coefs[1]*coefs[1]],
                     [ coors[0],  coors[1], coefs[0]*coefs[1]],
                     [-coors[0],  coors[0], coefs[0]*coefs[0]],
                     [ coors[1],  coors[0], coefs[1]*coefs[0]],
                     [ coors[0],  coors[0], coefs[0]*coefs[0]]])

    for igp in range(9):
        ksi = GSSP[igp, 0]
        eta = GSSP[igp, 1]

        tw_8_diff_ksi = np.array([
            0.25*(2*ksi + eta - 2*ksi*eta - eta**2),
            0.25*(2*ksi - eta + 2*ksi*eta - eta**2),
            0.25*(2*ksi + eta + 2*ksi*eta + eta**2),
            0.25*(2*ksi - eta - 2*ksi*eta + eta**2),
            0.50*(-1 + eta**2),
            0.50*(-2*ksi - 2*ksi*eta),
            0.50*( 1 - eta**2),
            0.50*(-2*ksi + 2*ksi*eta)])

        tw_8_diff_eta = np.array([
            0.25*(2*eta + ksi - 2*ksi*eta - ksi**2),
            0.25*(2*eta - ksi - 2*ksi*eta + ksi**2),
            0.25*(2*eta + ksi + 2*ksi*eta + ksi**2),
            0.25*(2*eta - ksi + 2*ksi*eta - ksi**2),
            0.50*(-2*eta + 2*ksi*eta),
            0.50*( 1 - ksi**2),
            0.50*(-2*eta - 2*ksi*eta),
            0.50*(-1 + ksi**2)])

        Bv     = np.vstack([tw_8_diff_ksi, tw_8_diff_eta])   # (2, 8)
        Jaco   = Bv @ np.column_stack([Node_X, Node_Y])       # (2, 2)
        DetJ   = np.linalg.det(Jaco)
        JaciBv = np.linalg.solve(Jaco, Bv)                    # (2, 8)
        factor = DetJ * GSSP[igp, 2]

        SJaciBv = S @ JaciBv
        k_ele  += factor * (SJaciBv.T @ Emat @ SJaciBv)

    return k_ele
