"""
Build CompactModel1 (Fourier basis + airgap tables) from CompactModel0.
"""

import time
import numpy as np

from umpfea.mdl.xfrm_2_Fourier import xfrm_2_Fourier
from umpfea.mdl.agap import calc_agapN
from umpfea.mdl.ModelBuild import CompactModel1


def macbuild1(compact_model0, Nh_S=25, Nh_R=25,
              N_ecc=20, xmax=0.95, Ncirc=None, Nlayer=3):
    """
    Build the Fourier-basis compact model.

    Parameters
    ----------
    compact_model0 : CompactModel0
    Nh_S, Nh_R : int
        Number of stator / rotor harmonics.
    N_ecc : int
        Number of eccentricity samples in the airgap table.
    xmax : float
        Largest tabulated eccentricity, as a fraction of the airgap.
    Ncirc : int or None
        Circumferential elements in the airgap mesh (None = default in calc_agap1).
    Nlayer : int
        Radial element layers in the airgap mesh.

    Returns
    -------
    CompactModel1
    """
    t_start = time.perf_counter()

    print('\n macbuild1 :: transforming stator matrices to Fourier basis ...')
    K_matSS_F, v_PM_S_F, T_mat_S_F = xfrm_2_Fourier(
        compact_model0.MNodes_S,
        Nh_S,
        compact_model0.K_matSS,
        compact_model0.v_PM_S,
        compact_model0.T_mat_S)

    print(' macbuild1 :: transforming rotor  matrices to Fourier basis ...')
    K_matRR_F, v_PM_R_F, T_mat_R_F = xfrm_2_Fourier(
        compact_model0.MNodes_R,
        Nh_R,
        compact_model0.K_matRR,
        compact_model0.v_PM_R,
        compact_model0.T_mat_R)

    # Mean radii of the master nodes
    Rad_S = np.mean(np.sqrt(compact_model0.MNodes_S[:, 0]**2
                            + compact_model0.MNodes_S[:, 1]**2))
    Rad_R = np.mean(np.sqrt(compact_model0.MNodes_R[:, 0]**2
                            + compact_model0.MNodes_R[:, 1]**2))

    print(f' macbuild1 :: Rad_R = {Rad_R*1000:.3f} mm,  Rad_S = {Rad_S*1000:.3f} mm')
    print(f'              airgap = {(Rad_S-Rad_R)*1000:.3f} mm')

    print(f'\n macbuild1 :: building airgap stiffness table'
          f'  (Nh_R={Nh_R}, Nh_S={Nh_S}, N_ecc={N_ecc}) ...')
    Eccs, KcosN, KsinN = calc_agapN(
        Rad_R, Rad_S, Nh_R, Nh_S, N_ecc, xmax, Ncirc, Nlayer)

    total_time = time.perf_counter() - t_start
    print(f'\n macbuild1 :: done. Total time = {total_time:.1f} s')

    return CompactModel1(
        MNodes_S  = compact_model0.MNodes_S,
        MNodes_R  = compact_model0.MNodes_R,
        K_matSS_F = K_matSS_F,
        v_PM_S_F  = v_PM_S_F,
        T_mat_S_F = T_mat_S_F,
        Nh_S      = Nh_S,
        Rad_S     = Rad_S,
        K_matRR_F = K_matRR_F,
        v_PM_R_F  = v_PM_R_F,
        T_mat_R_F = T_mat_R_F,
        Nh_R      = Nh_R,
        Rad_R     = Rad_R,
        Eccs      = Eccs,
        KcosN     = KcosN,
        KsinN     = KsinN,
        Lmac      = compact_model0.Lmac,
        NppR      = compact_model0.NppR,
    )
