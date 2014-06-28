import osmosis.emd as emd
import numpy as np
import numpy.testing as npt



def test_emd():
    xx = np.array([[1,0,0], [0,1,0], [0,0,1]])
    cost = []

    for this_x in xx:
        for this_y in xx:
            cost.append(np.sqrt(sum((this_x - this_y)**2)))

    ee = emd.emd([0, 1, 0], [1, 0, 0], list(cost))

    npt.assert_almost_equal(ee, np.sqrt(2), decimal=5)
