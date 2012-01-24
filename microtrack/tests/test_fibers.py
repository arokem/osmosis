
import microtrack.fibers as mtf

import numpy as np
import numpy.testing as npt
import numpy.testing.decorators as dec


def test_Fiber():
    """
    Test that initializing the fibers does something reasonable
    """
    f1 = mtf.Fiber(np.array([1,2,3]))
    f2 = mtf.Fiber(np.array([[1,2,3],[2,3,4]])
    
