import microtrack.tensor as mtt
import microtrack.fibers as mtf

import numpy as np
import numpy.testing as npt


def test_Tensor():
    """
    Test initialization of tensor objects
    """

    npt.assert_raises(ValueError,mtt.Tensor, np.arange(20))
    
    Q1 = [0,1,2,3,4,5]
    t1 = mtt.Tensor(Q1)
    
    # Follows conventions from vistasoft's dt6to33:
    npt.assert_equal(t1.Q, [[0,3,4],[3,1,5],[4,5,2]])

    # This should not be an acceptable input (would lead to asymmetric output):
    Q2 = [0,1,2,3,4,5,6,7,8]
    npt.assert_raises(ValueError, mtt.Tensor, Q2)

    # Same here: 
    Q3 = [[0,1,2],[3,4,5],[6,7,8]]
    npt.assert_raises(ValueError, mtt.Tensor, Q2)

    # No problem here:

    Q4 = [[0,3,4],[3,1,5],[4,5,2]]
    mtt.Tensor(Q4)

def test_Tensor_ADC():
    bvec = np.array([1,0,0])
    Q1 = np.eye(3)
    T1 = mtt.Tensor(Q1)
    npt.assert_equal(T1.ADC(bvec), 1)

    # This also works with multiple bvecs:
    bvecs = np.array([[0,0,1],[0,1,0]])
    npt.assert_equal(T1.ADC(bvecs), [1, 1])

# XXX This might need to be in some other place? 
def test_fiber_tensors():
    f1 = mtf.Fiber([[1,2],[3,4],[5,6]])

    # Values for axial and radial diffusivity
    ad = 1.5
    rd = 0.5
    mtt.fiber_tensors(f1, ad, rd)
    
    
