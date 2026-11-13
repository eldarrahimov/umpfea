"""
Reduced stator and rotor matrices for a machine.
"""

import numpy as np

from umpfea.mdl.objbuild import objbuild
from umpfea.mdl.Transforms import condense


def macbuild(machine_model):
    """
    Build the condensed stator and rotor matrices of a MachineModel.

    Each part is reduced with objbuild, then its CnstCmm nodes are merged
    into one common DOF, giving NMstr+1 DOFs per part. Results are per unit
    depth (Wb/m).

    Parameters
    ----------
    machine_model : MachineModel
        Stator and Rotor must already be dereferenced (deref_nls).

    Returns
    -------
    K_sttrC : (NMstrS+1, NMstrS+1) ndarray
        Stator stiffness; bore nodes first, common node last.
    v_sttrC : (NMstrS+1, 1) ndarray
    T_sttrC : (NMstrS+1, NCC_S) ndarray
    K_rotrC : (NMstrR+1, NMstrR+1) ndarray
    v_rotrC : (NMstrR+1, 1) ndarray
    T_rotrC : (NMstrR+1, NCC_R) ndarray

    Notes
    -----
    The back-substitution data returned by objbuild is discarded, so fields
    inside the parts cannot be recovered from these results.
    """
    sttr_model = machine_model.Stator
    rotr_model = machine_model.Rotor

    ndof_S = sttr_model.NodeSet.shape[0]
    ndof_R = rotr_model.NodeSet.shape[0]
    print('\n\n <macbuild> is addressing a machine in which there are')
    print(f' {ndof_S:5d} stator nodes and {ndof_R:5d} rotor nodes.')

    # Stator
    print('\n\n Developing a reduced numerical model of the stator\n')
    K_sttr, v_sttr, T_sttr, _, _, _, _ = objbuild(sttr_model, 'Stator')

    print('\n\n Now condensing out the common nodes from Stator\n')
    NMstrS = sttr_model.MstrNds.shape[0]
    K_sttrC, v_sttrC, T_sttrC = condense(K_sttr, v_sttr, T_sttr, NMstrS)

    # Rotor
    print('\n\n Developing a reduced numerical model of the rotor\n')
    K_rotr, v_rotr, T_rotr, _, _, _, _ = objbuild(rotr_model, 'Rotor')

    print('\n\n Now condensing out the common nodes from Rotor\n')
    NMstrR = rotr_model.MstrNds.shape[0]
    K_rotrC, v_rotrC, T_rotrC = condense(K_rotr, v_rotr, T_rotr, NMstrR)

    # Report asymmetry, then symmetrise
    asym_S = (np.max(np.abs(K_sttrC - K_sttrC.T)) /
              np.max(np.abs(K_sttrC + K_sttrC.T)))
    print(f' Extent of asymmetry in K_sttrC is {asym_S:.2g}')

    asym_R = (np.max(np.abs(K_rotrC - K_rotrC.T)) /
              np.max(np.abs(K_rotrC + K_rotrC.T)))
    print(f' Extent of asymmetry in K_rotrC is {asym_R:.2g}')

    K_sttrC = (K_sttrC + K_sttrC.T) / 2
    K_rotrC = (K_rotrC + K_rotrC.T) / 2

    # Lowest eigenvalues, as a sanity check
    eggs_sttr = np.linalg.eigvalsh(K_sttrC)
    eggs_rotr = np.linalg.eigvalsh(K_rotrC)
    n_show = min(6, len(eggs_sttr), len(eggs_rotr))
    for i in range(n_show):
        print(f' Eig #{i+1:2d} of K_sttrC is {eggs_sttr[i]:16.7g} : '
              f' Eig #{i+1:2d} of K_rotrC is {eggs_rotr[i]:16.7g}')

    return K_sttrC, v_sttrC, T_sttrC, K_rotrC, v_rotrC, T_rotrC
