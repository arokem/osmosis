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
    # If we have a diagonal tensor: 
    Q1_diag = np.random.rand(3)
    Q1 = np.diag(Q1_diag)
    T1 = mtt.Tensor(Q1)
    # And the bvecs are unit vectors: 
    bvecs = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # We should simply get back the diagonal elements of the tensor:
    npt.assert_equal(T1.ADC(bvecs), Q1_diag)

# XXX These might end up in some other place? 
def test_fiber_tensors():
    f1 = mtf.Fiber([[2,2,3],[3,3,4],[4,4,5]])
    # Values for axial and radial diffusivity randomly chosen:
    ad = np.random.rand()
    rd = np.random.rand()
    tensors = mtt.fiber_tensors(f1, ad, rd)
    npt.assert_equal(tensors[0].Q, np.diag([ad, rd, rd]))
    
def test_fiber_signal():
    f1 = mtf.Fiber([[2,2],[3,3],[4,4]])
    # Values for axial and radial diffusivity
    ad = 1.5
    rd = 0.5
    tensor_list = mtt.fiber_tensors(f1, ad, rd)

    # Simple bvecs/bvals:
    bvecs = [[1,0,0], [0,1,0], [1,0,0], [0,0,0]]
    bvals = [1,1,1,0]
    S0 = 1
    fs = mtt.fiber_signal(S0, bvecs, bvals, tensor_list)

    ind_nodes = [S0 * np.exp(-tensor_list[i].ADC(bvecs[0])) for i
                 in range(len(tensor_list))]
    # Summing over individual nodes: 
    sum_ind_nodes =  np.sum(ind_nodes)

    # Should give you the total signal:
    npt.assert_equal(fs[0],sum_ind_nodes)
