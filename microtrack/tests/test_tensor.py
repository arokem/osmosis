import microtrack.tensor as mtt

import numpy as np
import numpy.testing as npt


def test_Tensor():
    """
    Test initialization of tensor objects
    """

    npt.assert_raises(ValueError,mtt.Tensor, np.arange(20))
    
