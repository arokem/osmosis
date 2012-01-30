import os

import microtrack as mt
import microtrack.fibers as mtf

import numpy as np
import numpy.testing as npt
import numpy.testing.decorators as dec

import scipy.io as sio

def test_Fiber():
    """
    Testing initalization of the Fiber class
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
    f3 = mtf.Fiber(np.array(arr2d), affine = np.eye(4), fiber_stats=dict(a=1))
    npt.assert_equal(f3.fiber_stats, {'a':1})

def test_Fiber_xform():

    arr2d = np.array([[1,2,3],[2,3,4]])
    f1 = mtf.Fiber(arr2d)
    # XXX This is true, but not so good... 
    npt.assert_raises(NotImplementedError, f1.xform)

def test_FiberGroup():
    """
    Testing intialization of FiberGroup class.
    """
    
    arr2d = np.array([[1,2,3],[2,3,4]])
    arr1d = np.array([5,6,7])
    f1 = mtf.Fiber(arr2d, fiber_stats=dict(a=1, b=2))
    f2 = mtf.Fiber(arr1d, fiber_stats=dict(a=1))
    fg1 = mtf.FiberGroup([f1,f2])
    npt.assert_equal(fg1.n_fibers, 2)
    # We have to sort, because it could also come out as ['b', 'a']:
    npt.assert_equal(np.sort(fg1.fiber_stats.keys()), ['a', 'b'])

    
def test_read_from_pdb():
    data_path = os.path.split(mt.__file__)[0] + '/data/'
    file_name = data_path + "FG_w_stats.pdb"
    fg = mtf.fg_from_pdb(file_name)
    # Get the same fiber group as saved in matlab:
    mat_fg = sio.loadmat(data_path + "fg_from_matlab.mat",
                         squeeze_me=True)["fg"]
    k = [d[0] for d in mat_fg.dtype.descr]
    v = mat_fg.item()
    mat_fg_dict = dict(zip(k,v))
    npt.assert_equal(fg.name, mat_fg_dict["name"])
