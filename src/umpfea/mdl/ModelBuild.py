from dataclasses import dataclass
import numpy as np


@dataclass
class ObjectModel:
    NodeSet: np.ndarray = None
    Elems3: np.ndarray = None
    Elems4: np.ndarray = None
    MstrNds: np.ndarray = None
    CnstCmm: np.ndarray = None
    Bound: np.ndarray = None
    PDirs: np.ndarray = None
    MProps: np.ndarray = None
    CMaps: np.ndarray = None


@dataclass
class MachineModel:
    Stator: ObjectModel = None
    Rotor: ObjectModel = None
    Length: float = None
    NppR: int = None


@dataclass
class CompactModel0:
    """
    Nodal compact model (macbuild + condense).

    DOFs are the potentials at the airgap-facing nodes plus one common node.
    Flux quantities are per unit depth (Wb/m); multiply by Lmac for totals.
    """
    MNodes_S: np.ndarray = None   # (NMstrS, 2)  stator bore node coordinates (m)
    K_matSS:  np.ndarray = None   # (NMstrS+1, NMstrS+1)  stator stiffness
    v_PM_S:   np.ndarray = None   # (NMstrS+1, 1)  stator PM excitation
    T_mat_S:  np.ndarray = None   # (NMstrS+1, NCC_S)  stator circuit coupling
    MNodes_R: np.ndarray = None   # (NMstrR, 2)  rotor OD node coordinates (m)
    K_matRR:  np.ndarray = None   # (NMstrR+1, NMstrR+1)  rotor stiffness
    v_PM_R:   np.ndarray = None   # (NMstrR+1, 1)  rotor PM excitation
    T_mat_R:  np.ndarray = None   # (NMstrR+1, NCC_R)  rotor circuit coupling
    Lmac:     float = None        # machine active length (m)
    NppR:     int   = None        # number of rotor pole-pairs


@dataclass
class CompactModel1:
    """
    Fourier-basis compact model with tabulated airgap stiffness (macbuild1).

    Stator and rotor each have 2*Nh+2 DOFs:
      0          : DC
      1 .. Nh    : cos(ih*theta)
      Nh+1..2*Nh : sin(ih*theta)
      2*Nh+1     : common node

    The airgap stiffness is tabulated at N_ecc eccentricities and evaluated
    at run time by calc_agapi (interpolation) and calc_agapj (rotation).
    """
    MNodes_S  : np.ndarray = None   # (NMstrS, 2)  stator bore node coords (m)
    MNodes_R  : np.ndarray = None   # (NMstrR, 2)  rotor OD node coords (m)
    K_matSS_F : np.ndarray = None   # (2*Nh_S+2, 2*Nh_S+2)  stator Fourier stiffness
    v_PM_S_F  : np.ndarray = None   # (2*Nh_S+2, 1)  stator Fourier PM excitation
    T_mat_S_F : np.ndarray = None   # (2*Nh_S+2, NCC_S)  stator Fourier circuit coupling
    Nh_S      : int        = None   # number of stator harmonics
    Rad_S     : float      = None   # mean stator bore radius (m)
    K_matRR_F : np.ndarray = None   # (2*Nh_R+2, 2*Nh_R+2)  rotor Fourier stiffness
    v_PM_R_F  : np.ndarray = None   # (2*Nh_R+2, 1)  rotor Fourier PM excitation
    T_mat_R_F : np.ndarray = None   # (2*Nh_R+2, NCC_R)  rotor Fourier circuit coupling
    Nh_R      : int        = None   # number of rotor harmonics
    Rad_R     : float      = None   # mean rotor OD radius (m)
    Eccs      : np.ndarray = None   # (N_ecc,)  eccentricity sample values (m)
    KcosN     : np.ndarray = None   # (Nh_R+Nh_S+2, Nh_R+Nh_S+2, N_ecc)
    KsinN     : np.ndarray = None   # (Nh_R+Nh_S,   Nh_R+Nh_S,   N_ecc)
    Lmac      : float      = None   # machine active length (m)
    NppR      : int        = None   # number of rotor pole-pairs
