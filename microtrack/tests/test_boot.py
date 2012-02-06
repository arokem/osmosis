import numpy as np
import numpy.testing as npt

import microtrack.boot as mtb


def test_subsample():
    """
    Test subsampling
    """

    # Sub-sampling 100 out of a random collection of 150 unit-vectors:
    bvecs = np.array([mtb.unit_vector(x) for x in np.random.randn(3,150)]).T

    # The following runs through most of the module w/o verifying correctness:
    sub_sample = mtb.subsample(bvecs, 100)
