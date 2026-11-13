import numpy as np

def findlabl(objects, labels):
    """
    Row indices of `labels` in column 0 of `objects`, -1 where not found.

    A 1-D `labels` is treated as a single row, so the result is always 2-D.
    """
    labels = np.atleast_2d(np.asarray(labels))
    indices = np.zeros_like(labels, dtype=int)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            diff = np.abs(objects[:, 0] - labels[i, j])
            a = diff.min()
            b = diff.argmin()
            indices[i, j] = b if a == 0 else -1

    return indices


def bisect_KPs(KeyPoints, KP_Labl1, KP_Labl2, KP_Labl0=0):
    """
    Mid-point of the line or arc between two KeyPoints.

    KP_Labl0 is the arc centre; 0 means a straight line. If the centre is
    not equidistant from the two end points it is moved the minimum amount
    needed to make it so.

    Parameters
    ----------
    KeyPoints : (N, 3+) ndarray
        [label, x, y, ...]
    KP_Labl1, KP_Labl2 : end point labels
    KP_Labl0 : centre label, or 0 for a straight line

    Returns
    -------
    KP_xy : (2,) ndarray
        Mid-point.
    c_xy  : (2,) ndarray
        Arc centre (equal to KP_xy for a straight line).
    """

    if abs(KP_Labl0) > 0:          # Circular arc
        labls = [KP_Labl1, KP_Labl2, KP_Labl0]
        indxs = findlabl(KeyPoints, labls).ravel()

        if indxs.min() < 0:
            raise ValueError(
                f"ERROR in bisect_KPs: not all labels found.\n"
                f"Labels: {labls}\nIndices: {indxs}"
            )

        pts_xy = KeyPoints[indxs, 1:3]

        # Move the centre onto the perpendicular bisector of pt1-pt2
        vct12  = (pts_xy[1] - pts_xy[0])
        vctmp  = (pts_xy[1] + pts_xy[0]) / 2
        cntr0  = pts_xy[2].copy()

        vctad  = vct12 * (vct12 @ (cntr0 - vctmp)) / (vct12 @ vct12)
        cntrc  = cntr0 - vctad

        # Project the chord mid-point out to the arc
        vctc1  = pts_xy[0] - cntrc
        vctc2  = pts_xy[1] - cntrc
        vctcm  = vctmp    - cntrc

        rad1   = np.sqrt(vctc1 @ vctc1)
        rad2   = np.sqrt(vctc2 @ vctc2)
        radm   = (rad1 + rad2) / 2

        tx = np.sqrt(vctcm @ vctcm)

        if tx > 1e-6 * radm:
            vctcp = vctcm * radm / tx
            KP_xy = cntrc + vctcp
        else:
            KP_xy = (pts_xy[0] + pts_xy[1]) / 2

        c_xy = cntrc

    else:                           # Straight line
        labls = [KP_Labl1, KP_Labl2]
        indxs = findlabl(KeyPoints, labls).ravel()

        if indxs.min() < 0:
            raise ValueError(
                f"ERROR in bisect_KPs: not all labels found.\n"
                f"Labels: {labls}\nIndices: {indxs}"
            )

        pts_xy = KeyPoints[indxs, 1:3]
        KP_xy  = pts_xy.mean(axis=0)
        c_xy   = KP_xy.copy()

    return KP_xy, c_xy


def bisect_xy(KeyPoints, a, b, c=0):
    return bisect_KPs(KeyPoints, a, b, c)[0]
