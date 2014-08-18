import numpy as np
import numpy.testing as npt

import osmosis.emd as emd
import osmosis.utils as ozu
import osmosis.predict_n as pn



def test_emd():
    xx = np.array([[1,0,0], [0,1,0], [0,0,1]])
    cost = []

    for this_x in xx:
        for this_y in xx:
            cost.append(np.sqrt(sum((this_x - this_y)**2)))

    ee = emd.emd([0, 1, 0], [1, 0, 0], list(cost))
    npt.assert_almost_equal(ee, np.sqrt(2), decimal=5)


def test_fodf_emd():
    """
    Test EMD on fODFs
    """ 
    bvecs = ozu.get_camino_pts(150)
    fodf1 = np.zeros(bvecs.shape[-1])
    fodf2 = np.zeros(bvecs.shape[-1])
    ii = np.random.randint(0, fodf1.shape[0])
    jj = np.random.randint(0, fodf2.shape[0])
    fodf1[ii] = 1
    fodf2[jj] = 1

    emd1 = pn.fODF_EMD(fodf1, fodf2, bvecs1=bvecs, bvecs2=bvecs)

    angles = np.arccos(np.dot(bvecs.T, bvecs))
    angles[np.isnan(angles)] = 0
    angles = np.min(np.array([angles, np.pi-angles]),0)
    angles = angles.ravel()
    emd2 = pn.fODF_EMD(fodf1, fodf2, bvecs1=bvecs, dist=angles)

    npt.assert_equal(emd1, emd2)
