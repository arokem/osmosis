
import microtrack.fibers as mtf

import numpy as np
import numpy.testing as npt
import numpy.testing.decorators as dec


def test_Fiber():
    """
    Test that initializing the fibers does something reasonable
    """
    arr1d = np.array([1,2,3])
    # This is the most basic example possible:
    f1 = mtf.Fiber(arr1d)
    # 2D arrays should be n by 3:
    arr2d = np.array([[1,2,3],[2,3,4]])
    # So this is OK:
    f2 = mtf.Fiber(arr2d)    
    # But this raises a ValueError:
    npt.assert_raises(ValueError, mtf.Fiber, arr2d.T)
    # This should also raise (second dim is 4, rather than 3): 
    npt.assert_raises(ValueError, mtf.Fiber, np.empty((10,4)))
    # This should be OK:
    f3 = mtf.Fiber(np.array(arr2d), affine = np.eye(4), stats=dict(a=1))
    npt.assert_equal(f3.stats, {'a':1})
    # Stats should make sense (this one is not one per node and not one for the
    # entire fiber): 
    npt.assert_raises(ValueError, mtf.Fiber, arr2d, np.eye(4),
                                              dict(a=np.array([1,2])))

def test_Fiber_xform():

    arr2d = np.array([[1,2,3],[2,3,4]])
    f1 = mtf.Fiber(arr2d)
    # XXX This is true, but not so good... 
    npt.assert_raises(NotImplementedError, f1.xform)
    
def test_read_from_pdb():
    file_name = "/Users/arokem/source/microtrack/data/FG_w_stats.pdb"
    out = mtf.fg_from_pdb(file_name)
    npt.assert_equal(out, (3276, np.eye(4), 3, 7))
