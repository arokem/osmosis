import numpy as np
import numpy.testing as npt

import osmosis as mt
import osmosis.boot as mtb
import osmosis.utils as mtu

def test_subsample():
    """
    Test subsampling
    """

    # Sub-sampling 100 out of a random collection of 150 unit-vectors:
    bvecs = np.array([mtu.unit_vector(x) for x in np.random.randn(3,150)]).T

    # The following runs through most of the module w/o verifying correctness:
    sub_sample = mtb.subsample(bvecs, 100)

    # optionally, you can provide elec_points as input. Here we test this with
    # the same points
    mt_path = mt.__path__[0]
    e_points = np.loadtxt('%s/camino_pts/Elec%03d.txt'%(mt_path, 100))
    sub_sample = mtb.subsample(bvecs, 100, elec_points=e_points)
    
def test_dyad():
    """
    Test the dyadic tensor and coherence based on the dyadic tensor
    """

    eigs1 = [np.eye(3), np.eye(3)]
    dyad = mtb.dyadic_tensor(eigs1, average=True)

    npt.assert_equal(mtb.dyad_coherence(dyad), 1.0)

    # For this collection of eigenvectors: 
    eigs2 = np.array([[[1,0,0],[0,2,0],[0,0,3]],[[1,0,0],[0,2,0],[0,0,3]]])
    dyad_dist = mtb.dyadic_tensor(eigs2, average=False)

    # The dispersion should be null:
    npt.assert_equal(mtb.dyad_dispersion(dyad_dist), 0)
