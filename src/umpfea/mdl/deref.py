import numpy as np
from umpfea.geom.labls import findlabl
from umpfea.mdl.ModelBuild import ObjectModel


def _deref(NodeSet, labels, what):
    indxs = findlabl(NodeSet, labels).ravel()
    if indxs.size and indxs.min() < 0:
        bad = np.asarray(labels).ravel()[indxs < 0]
        raise ValueError(f'deref_nls: unknown node label(s) {bad} in {what}')
    return indxs


def deref_nls(Model):
    """
    Replace node labels by 0-based NodeSet row indices.

    Applies to element topologies, Bound, CnstCmm and MstrNds. The label
    column is dropped from NodeSet. PD/Mat/CMap labels are left as they are.

    Parameters
    ----------
    Model : ObjectModel
        Label-based part model.

    Returns
    -------
    Model_DR : ObjectModel
        Index-based copy.
    """
    nele3     = Model.Elems3.shape[0]
    Elems3_DR = np.zeros((nele3, 9))
    for iele3 in range(nele3):
        rowt  = Model.Elems3[iele3, :]
        indxs = _deref(Model.NodeSet, rowt[0:6], f'Elems3 row {iele3}')
        Elems3_DR[iele3, :] = np.concatenate((indxs, rowt[6:]))

    nele4     = Model.Elems4.shape[0]
    Elems4_DR = np.zeros((nele4, 11))
    for iele4 in range(nele4):
        rowt  = Model.Elems4[iele4, :]
        indxs = _deref(Model.NodeSet, rowt[0:8], f'Elems4 row {iele4}')
        Elems4_DR[iele4, :] = np.concatenate((indxs, rowt[8:]))

    nbound   = Model.Bound.shape[0]
    Bound_DR = np.zeros((nbound, 2))
    for ibnd in range(nbound):
        rowt  = Model.Bound[ibnd, :]
        Bound_DR[ibnd, :] = _deref(Model.NodeSet, rowt[0:2], f'Bound row {ibnd}')

    CnstCmm_DR = _deref(Model.NodeSet, Model.CnstCmm, 'CnstCmm')
    MstrNds_DR = _deref(Model.NodeSet, Model.MstrNds, 'MstrNds')

    NodeSet_DR = Model.NodeSet[:, 1:]

    Model_DR          = ObjectModel()
    Model_DR.NodeSet  = NodeSet_DR
    Model_DR.Elems3   = Elems3_DR
    Model_DR.Elems4   = Elems4_DR
    Model_DR.Bound    = Bound_DR
    Model_DR.CnstCmm  = CnstCmm_DR
    Model_DR.MstrNds  = MstrNds_DR
    Model_DR.CMaps    = Model.CMaps
    Model_DR.MProps   = Model.MProps
    Model_DR.PDirs    = Model.PDirs

    return Model_DR
