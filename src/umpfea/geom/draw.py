"""
Plotting of model meshes.

plotelems / plotbound take label-based arrays (NodeSet = [label, x, y]).
draw_obj takes a dereferenced model from deref_nls (NodeSet = [x, y],
0-based node indices).
"""

import numpy as np
import matplotlib.pyplot as plt
from .labls import findlabl


def plotelems(NodeSet, Elems3, Elems4, MProps=None, ax=None):
    """
    Plot element outlines, filled with the material colour if MProps is given.

    Parameters
    ----------
    NodeSet : (N, 3) ndarray   [label, x, y]
    Elems3  : (M, 9) ndarray
    Elems4  : (P, 11) ndarray
    MProps  : ndarray or None  material table, RGB in columns 5:8
    ax      : matplotlib Axes or None (default: current axes)
    """
    if ax is None:
        ax = plt.gca()

    # Outline order: corner, mid-side, corner, ...
    seq3 = np.array([0, 3, 1, 4, 2, 5, 0])
    for iele in range(Elems3.shape[0]):
        indxs  = findlabl(NodeSet, Elems3[iele, 0:6]).ravel()
        coords = NodeSet[indxs[seq3], 1:3]
        if MProps is not None:
            mindx = findlabl(MProps, Elems3[iele, 7]).ravel()[0]
            color = MProps[mindx, 5:8]
            ax.fill(coords[:-1, 0], coords[:-1, 1], color=color, edgecolor='none')
        ax.plot(coords[:, 0], coords[:, 1], color='k', linewidth=0.5, alpha=0.5,
                solid_capstyle='butt')

    seq4 = np.array([0, 4, 1, 5, 2, 6, 3, 7, 0])
    for iele in range(Elems4.shape[0]):
        indxs  = findlabl(NodeSet, Elems4[iele, 0:8]).ravel()
        coords = NodeSet[indxs[seq4], 1:3]
        if MProps is not None:
            mindx = findlabl(MProps, Elems4[iele, 9]).ravel()[0]
            color = MProps[mindx, 5:8]
            ax.fill(coords[:-1, 0], coords[:-1, 1], color=color, edgecolor='none')
        ax.plot(coords[:, 0], coords[:, 1], color='k', linewidth=0.5, alpha=0.5,
                solid_capstyle='butt')


def plotbound(NodeSet, Bound, ax=None):
    """
    Plot boundary segments as thick black lines.

    Parameters
    ----------
    NodeSet : (N, 3) ndarray  [label, x, y]
    Bound   : (Q, 2) ndarray  node label pairs
    ax      : matplotlib Axes or None (default: current axes)
    """
    if ax is None:
        ax = plt.gca()

    for ibnd in range(Bound.shape[0]):
        indxs  = findlabl(NodeSet, Bound[ibnd, :]).ravel()
        coords = NodeSet[indxs, 1:3]
        ax.plot(coords[:, 0], coords[:, 1], 'k', linewidth=2, solid_capstyle='butt')


def draw_obj(model, title_str='', iopt=0, ax=None, figsize=(8, 8), dpi=150):
    """
    Draw a dereferenced part model: nodes, elements, boundary and optionally
    the commoned / master nodes.

    Parameters
    ----------
    model     : ObjectModel from deref_nls
    title_str : str, axes title
    iopt      : int, bit flags
                  1: mark CnstCmm nodes (green)
                  2: mark MstrNds nodes (red)
    ax        : matplotlib Axes or None (new figure)
    figsize, dpi : used only when a new figure is created

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    NodeSet = model.NodeSet   # (N, 2) [x, y]
    Elems3  = model.Elems3
    Elems4  = model.Elems4
    Bound   = model.Bound
    MProps  = model.MProps
    CnstCmm = model.CnstCmm
    MstrNds = model.MstrNds

    print(f' <draw_obj> ... Number of Nodes   = {NodeSet.shape[0]:6d}')
    print(f' <draw_obj> ... Number of Elem3s  = {Elems3.shape[0]:6d}')
    print(f' <draw_obj> ... Number of Elem4s  = {Elems4.shape[0]:6d}')
    print(f' <draw_obj> ... Num of Bound Segs = {Bound.shape[0]:6d}')

    idraw_Cmm = iopt & 1
    idraw_Mst = (iopt >> 1) & 1

    ax.plot(NodeSet[:, 0], NodeSet[:, 1], 'k.', markersize=2)
    ax.set_aspect('equal')
    ax.axis('off')
    if title_str:
        ax.set_title(title_str)

    # Topology is index-based here; material labels are still labels
    seq3 = np.array([0, 3, 1, 4, 2, 5, 0])
    for iele in range(Elems3.shape[0]):
        indxs  = Elems3[iele, 0:6].astype(int)
        coords = NodeSet[indxs[seq3], 0:2]
        if MProps is not None:
            mindx = findlabl(MProps, Elems3[iele, 7]).ravel()[0]
            color = MProps[mindx, 5:8]
            ax.fill(coords[:-1, 0], coords[:-1, 1], color=color, edgecolor='none')
        ax.plot(coords[:, 0], coords[:, 1], color='k', linewidth=0.6, alpha=1,
                solid_capstyle='butt')

    seq4 = np.array([0, 4, 1, 5, 2, 6, 3, 7, 0])
    for iele in range(Elems4.shape[0]):
        indxs  = Elems4[iele, 0:8].astype(int)
        coords = NodeSet[indxs[seq4], 0:2]
        if MProps is not None:
            mindx = findlabl(MProps, Elems4[iele, 9]).ravel()[0]
            color = MProps[mindx, 5:8]
            ax.fill(coords[:-1, 0], coords[:-1, 1], color=color, edgecolor='none')
        ax.plot(coords[:, 0], coords[:, 1], color='k', linewidth=0.6, alpha=1,
                solid_capstyle='butt')

    for ibnd in range(Bound.shape[0]):
        indxs  = Bound[ibnd, :].astype(int)
        coords = NodeSet[indxs, 0:2]
        ax.plot(coords[:, 0], coords[:, 1], 'k', linewidth=2, solid_capstyle='butt')

    if idraw_Cmm:
        cmm = np.asarray(CnstCmm).ravel().astype(int)
        if cmm.size > 0:
            ax.plot(NodeSet[cmm, 0], NodeSet[cmm, 1],
                    'go', markersize=6, fillstyle='none', label='constrained')

    if idraw_Mst:
        mst = np.asarray(MstrNds).ravel().astype(int)
        if mst.size > 0:
            ax.plot(NodeSet[mst, 0], NodeSet[mst, 1],
                    'ro', markersize=6, fillstyle='none', label='master')

    return ax
