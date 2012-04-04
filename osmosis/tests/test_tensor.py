import exceptions

import osmosis.tensor as mtt
import osmosis.fibers as mtf

import numpy as np
import numpy.testing as npt


def test_Tensor():
    """
    Test initialization of tensor objects
    """
    bvecs = [[1,0,0],[0,1,0],[0,0,1]]
    bvals = [1,1,1]
    
    npt.assert_raises(ValueError,mtt.Tensor, np.arange(20), bvecs, bvals)
    
    Q1 = [0,1,2,3,4,5]
    t1 = mtt.Tensor(Q1, bvecs, bvals)
    
    # Follows conventions from vistasoft's dt6to33:
    npt.assert_equal(t1.Q, [[0,3,4],[3,1,5],[4,5,2]])

    # This should not be an acceptable input (would lead to asymmetric output):
    Q2 = [0,1,2,3,4,5,6,7,8]
    npt.assert_raises(ValueError, mtt.Tensor, Q2, bvecs, bvals)

    # Same here: 
    Q3 = [[0,1,2],[3,4,5],[6,7,8]]
    npt.assert_raises(ValueError, mtt.Tensor, Q2, bvecs, bvals)

    # No problem here:
    Q4 = [[0,3,4],[3,1,5],[4,5,2]]
    mtt.Tensor(Q4,bvecs,bvals)

    # More error handling:
    # bvecs and bvals need to be the same length:
    Q5 = npt.assert_raises(ValueError, mtt.Tensor, Q4, bvecs[:2], bvals)

    # bvecs needs to be 3 by n:
    Q6 = npt.assert_raises(ValueError, mtt.Tensor, Q4, [[1,0,0,0],
                                                        [0,1,0,0],
                                                        [0,0,1,0],
                                                        [0,0,0,1]],
        bvals)

    # bvecs need to be unit length!
    Q7 = npt.assert_raises(ValueError, mtt.Tensor, Q4, [[1,0,0],[0,1,0],[0,1,1]],
                           bvals)
    

    # bvecs and bvals need to have the same length!
    Q8 = npt.assert_raises(ValueError, mtt.Tensor, Q4, [[1,0,0],[0,1,0],[0,0,1]],
                           np.hstack([bvals, 1]))


def test_Tensor_ADC():
    # If we have a diagonal tensor: 
    Q1_diag = np.random.rand(3)
    Q1 = np.diag(Q1_diag)

    # And the bvecs are unit vectors: 
    bvecs = np.array([[1,0,0],[0,1,0],[0,0,1]])
    bvals = [1,1,1]
    # Initialize: 
    T1 = mtt.Tensor(Q1,bvecs,bvals)
    # We should simply get back the diagonal elements of the tensor:
    npt.assert_equal(T1.ADC, Q1_diag)

def test_Tensor_predicted_signal():
    """
    Test prediction of the signal from the Tensor:
    """
    Q1_diag = np.random.rand(3)
    Q1 = np.diag(Q1_diag)

    # And the bvecs are unit vectors: 
    bvecs = np.array([[1,0,0],[0,1,0],[0,0,1]])
    bvals = [1,1,1]

    T1 = mtt.Tensor(Q1, bvecs, bvals)

    # Should be possible to provide a value of S0 per bvec:
    S0_1 = [1,1,1]

    # In this case, that should give the same answer as just providing a single
    # scalar value:
    S0_2 = [1]
    
    npt.assert_equal(T1.predicted_signal(S0_1),
                     T1.predicted_signal(S0_2))
                     

def test_Tensor_decompose():
    """
    Test the eigen-vector/value decomposition of the tensor:
    """
    q = np.array([[1.5,0,0], 
                  [0,0.51,0],  # This needs to be slightly higher, so that
                               # there is no ambiguity about the order of the
                               # evals 
                  [0,0,0.5]])

    # And the bvecs are unit vectors: 
    bvecs = np.array([[1,0,0],[0,1,0],[0,0,1]])
    bvals = [1,1,1]

    T1 = mtt.Tensor(q, bvecs, bvals)
    vals, vecs = T1.decompose

    npt.assert_equal(vecs , np.eye(3))
    npt.assert_equal(vals, np.diag(q))

    T2 = mtt.tensor_from_eigs(vecs, vals, bvecs, bvals)

    npt.assert_equal(T2.Q,q)
    
def test_convlove_odf():
    """
    Test convolution of a tensor with a fiber orientation distribution function
    (odf)
    """ 
    Q1_diag = np.random.rand(3)
    Q1 = np.diag(Q1_diag)

    # And the bvecs are unit vectors (one's taken from actual data, so that we
    # have a 3 by 4): 
    bvecs = np.array([[1,0,0],[0,1,0],[0,0,1],[-0.24187,  0.10309, -0.96482]]).T
    bvals = [1,1,1,1]

    T1 = mtt.Tensor(Q1, bvecs, bvals)

    S0 = 1000

    # Has the same length as the number of bvecs:
    odf = [0.1,0.2,0.3,0.4]

    # This performs the convolution
    T1.convolve_odf(odf, S0)

def test_stejskal_tanner():
    """
    Test the implementation of the S/T equation.

    """
    bvecs = [[1,0,0],[0,1,0],[0,0,1]]
    bvals = [1,1,1]
    S0 = 1
    ADC = np.random.rand(3)
    Q = np.diag(ADC)

    # Calculate explicitely:
    st = S0 * np.exp(-1 * np.asarray(bvecs).T * Q * np.asarray(bvecs))

    # These should give the same answer: 
    npt.assert_equal(mtt.stejskal_tanner(S0, bvals, ADC), np.diag(st))

    
