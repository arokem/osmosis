import numpy as np
import numpy.testing as npt
import scipy.linalg as la

import microtrack.utils as mtu

def test_unique_coords():
    """
    Testing the function unique_coords
    """
    arr = np.array([[1,2,3],[1,2,3],[2,3,4],[3,4,5]])
    arr_w_unique = np.array([[1,2,3],[2,3,4],[3,4,5]])
    npt.assert_equal(mtu.unique_rows(arr), arr_w_unique)

    # Should preserve order:
    arr = np.array([[2,3,4],[1,2,3],[1,2,3],[3,4,5]])
    arr_w_unique = np.array([[2,3,4],[1,2,3],[3,4,5]])
    npt.assert_equal(mtu.unique_rows(arr), arr_w_unique)

    
    # Should work even with longer arrays:
    arr = np.array([[2,3,4],[1,2,3],[1,2,3],[3,4,5],
                    [6,7,8],[0,1,0],[1,0,1]])
    arr_w_unique = np.array([[2,3,4],[1,2,3],[3,4,5],
                             [6,7,8],[0,1,0],[1,0,1]])
    
    npt.assert_equal(mtu.unique_rows(arr), arr_w_unique)

def test_intersect():
    """
    Testing the multi-intersect utility function
    """

    arr1 = np.array(np.arange(1000).reshape(2,500))
    arr2 = np.array([[1,0.1,0.2],[0.3,0.4, 0.5]])
    arr3 = np.array(1)
    npt.assert_equal(1, mtu.intersect([arr1, arr2, arr3]))

def test_euclidian_distance():
    """
    Testing the euclidian distance metrix
    """
    x = [np.array([1,2,3])]
    y = [np.array([1,2,3])]

    npt.assert_equal(mtu.euclidian_distance(x,y),0)


    x = [np.array([0,0,1]), np.array([0,1,0])]
    y = [np.array([0,0,0]), np.array([0,0,0])]

    npt.assert_equal(mtu.euclidian_distance(x,y), np.ones((2,2)))

    y = [np.array([0,1,0]), np.array([0,0,1])]
    npt.assert_equal(mtu.euclidian_distance(x,y), np.sqrt(2) * np.eye(2))
    
def test_decompose_tensor():
    """
    Testing the decomposition and recomposition of tensors from eigen-vectors
    and eigen-values

    """

    # This is the self-diffusion tensor of water (according to Basser et
    # al. 1994):
    q1 = np.array([[1.7003,    -0.041,    0.0027], 
                   [-0.041,    1.6388,    -0.0036], 
                   [0.0027,    -0.0036,    1.7007]])

    evals1, evecs1 = mtu.decompose_tensor(q1)

    q2 = mtu.tensor_from_eigs(evals1, evecs1)

    npt.assert_almost_equal(q1, q2)

    evals2, evecs2 = mtu.decompose_tensor(q2)
    
    q3 = mtu.tensor_from_eigs(evals1, evecs2)

    npt.assert_almost_equal(q2,q3)

    # Let's see that we can do this many times: 
    for i in range(1000):
        print i
        A = np.random.rand(9).reshape(3,3)

        # Make a symmetrical matrix (a 'tensor'):
        T1 = np.dot(A, A.T)
        evals, evecs = mtu.decompose_tensor(T1)
        
        Q = mtu.tensor_from_eigs(evals, evecs)
        evals_est, evecs_est = mtu.decompose_tensor(Q)

        # The result is always going to be equal, up to a sign reversal of one
        # of the vectors (why does this happen?):
        npt.assert_almost_equal(np.abs(evecs_est/evecs), np.ones((3,3)))
    
    
def test_ols_matrix():
    """
    Test that this really does OLS regression.
    """
    # Parameters
    beta = np.random.rand(10)
    # Inputs
    x = np.random.rand(100,10)
    # Outputs (noise-less!)
    y = np.dot(x, beta)
    # Estimate back:
    beta_hat = np.array(np.dot(mtu.ols_matrix(x),y)).squeeze()
    # This should have recovered the original:
    npt.assert_almost_equal(beta, beta_hat)
