import numpy as np
import numpy.testing as npt

import osmosis as oz
import osmosis.boot as ozb
import osmosis.utils as ozu

def test_subsample():
    """
    Test subsampling
    """

    # Sub-sampling 100 out of a random collection of 150 unit-vectors:
    bvecs = np.array([ozu.unit_vector(x) for x in np.random.randn(3,150)]).T

    # The following runs through most of the module w/o verifying correctness:
    sub_sample = ozb.subsample(bvecs, 100)

    # optionally, you can provide elec_points as input. Here we test this with
    # the same points
    sub_sample = ozb.subsample(bvecs, 100, elec_points=ozu.get_camino_pts(100).T)
    
def test_dyad():
    """
    Test the dyadic tensor and coherence based on the dyadic tensor
    """

    eigs1 = [np.eye(3), np.eye(3)]
    dyad = ozb.dyadic_tensor(eigs1, average=True)

    npt.assert_equal(ozb.dyad_coherence(dyad), 1.0)

    # For this collection of eigenvectors: 
    eigs2 = np.array([[[1,0,0],[0,2,0],[0,0,3]],[[1,0,0],[0,2,0],[0,0,3]]])
    dyad_dist = ozb.dyadic_tensor(eigs2, average=False)

    # The dispersion should be null:
    npt.assert_equal(ozb.dyad_dispersion(dyad_dist), 0)
