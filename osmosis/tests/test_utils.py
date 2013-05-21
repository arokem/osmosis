import numpy as np
import numpy.testing as npt
import scipy.linalg as la

import osmosis.utils as ozu

def test_unique_coords():
    """
    Testing the function unique_coords
    """
    arr = np.array([[1,2,3],[1,2,3],[2,3,4],[3,4,5]])
    arr_w_unique = np.array([[1,2,3],[2,3,4],[3,4,5]])
    npt.assert_equal(ozu.unique_rows(arr), arr_w_unique)

    # Should preserve order:
    arr = np.array([[2,3,4],[1,2,3],[1,2,3],[3,4,5]])
    arr_w_unique = np.array([[2,3,4],[1,2,3],[3,4,5]])
    npt.assert_equal(ozu.unique_rows(arr), arr_w_unique)

    
    # Should work even with longer arrays:
    arr = np.array([[2,3,4],[1,2,3],[1,2,3],[3,4,5],
                    [6,7,8],[0,1,0],[1,0,1]])
    arr_w_unique = np.array([[2,3,4],[1,2,3],[3,4,5],
                             [6,7,8],[0,1,0],[1,0,1]])
    
    npt.assert_equal(ozu.unique_rows(arr), arr_w_unique)

def test_intersect():
    """
    Testing the multi-intersect utility function
    """

    arr1 = np.array(np.arange(1000).reshape(2,500))
    arr2 = np.array([[1,0.1,0.2],[0.3,0.4, 0.5]])
    arr3 = np.array(1)
    npt.assert_equal(1, ozu.intersect([arr1, arr2, arr3]))

def test_euclidian_distance():
    """
    Testing the euclidian distance metrix
    """
    x = [np.array([1,2,3])]
    y = [np.array([1,2,3])]

    npt.assert_equal(ozu.euclidian_distance(x,y),0)


    x = [np.array([0,0,1]), np.array([0,1,0])]
    y = [np.array([0,0,0]), np.array([0,0,0])]

    npt.assert_equal(ozu.euclidian_distance(x,y), np.ones((2,2)))

    y = [np.array([0,1,0]), np.array([0,0,1])]
    npt.assert_equal(ozu.euclidian_distance(x,y), np.sqrt(2) * np.eye(2))


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

    evals1, evecs1 = ozu.decompose_tensor(q1)

    q2 = ozu.tensor_from_eigs(evals1, evecs1)

    npt.assert_almost_equal(q1, q2, decimal=2)

    evals2, evecs2 = ozu.decompose_tensor(q2)
    
    q3 = ozu.tensor_from_eigs(evals1, evecs2)

    npt.assert_almost_equal(q2,q3, decimal=2)

    # Let's see that we can do this many times: 
    for i in range(1000):
        print i
        A = np.random.rand(9).reshape(3,3)

        # Make a symmetrical matrix (a 'tensor'):
        T1 = np.dot(A, A.T)
        evals, evecs = ozu.decompose_tensor(T1)
        
        Q = ozu.tensor_from_eigs(evals, evecs)
        evals_est, evecs_est = ozu.decompose_tensor(Q)

        # The result is always going to be equal, up to a sign reversal of one
        # of the vectors (why does this happen?):
        npt.assert_almost_equal(np.abs(evecs_est), np.abs(evecs), decimal=3)


def test_fractional_anisotropy():
    """
    Test the calculation of FA
    """
    # Test this both with and without numexpr, if you can: 
    try:
        import numexpr
        has_numexpr = True
    except ImportError:
        has_numexpr = False

    if has_numexpr:
        for tst_numexpr in [False,True]:
            ozu.has_numexpr = tst_numexpr
            npt.assert_almost_equal(ozu.fractional_anisotropy(1,0,0), 1)
            npt.assert_almost_equal(ozu.fractional_anisotropy(1,1,1), 0)
    else:
            npt.assert_almost_equal(ozu.fractional_anisotropy(1,0,0), 1)
            npt.assert_almost_equal(ozu.fractional_anisotropy(1,1,1), 0)

    
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
    ols_matrix = ozu.ols_matrix(x)
    beta_hat = np.array(np.dot(ols_matrix, y)).squeeze()
    # This should have recovered the original:
    npt.assert_almost_equal(beta, beta_hat)
    
    # Make sure that you can normalize and it gives you the same shape matrix: 
    npt.assert_almost_equal(ols_matrix.shape,
                            ozu.ols_matrix(x, norm_func=ozu.l2_norm).shape)

def test_vector_angle():
    """

    Test that calculation of angles between vectors makes sense. 

    """

    a = [1, 0, 0]
    b = [0, 0, 1]

    npt.assert_equal(ozu.vector_angle(a,b), np.pi/2)

    a = [1,0,0]
    b = [-1,0,0]
    
    npt.assert_equal(ozu.vector_angle(a,b), np.pi)

    a = [1,0,0]
    b = [1,0,0]
    
    npt.assert_equal(ozu.vector_angle(a,b), 0)


def test_coeff_of_determination():
    """
    Test the calculation of the coefficient of determination
    """
    # These are two corner cases that should lead to a nan answer:
    data = np.zeros(10)
    model = np.zeros(10)
    npt.assert_equal(np.isnan(ozu.coeff_of_determination(data,model)),
                              True)


    data = np.zeros(10)
    model = np.random.randn(10)
    npt.assert_equal(np.isnan(ozu.coeff_of_determination(data,model)),
                              True)

    # This should be perfect:
    data = np.random.randn(10)
    model = np.copy(data)
    npt.assert_equal(ozu.coeff_of_determination(data,model), 1)

    

def test_rescale():
    """
    Test rescaling of data into [0,1]
    """

    data = np.random.randn(20) * 100
    rs = ozu.rescale(data)
    npt.assert_equal(np.max(rs),1)
    npt.assert_equal(np.min(rs),0)

    # Test for conditions in which the minimum is >0:
    data = (np.random.rand(20) + 10) * 100
    rs = ozu.rescale(data)
    npt.assert_equal(np.max(rs),1)
    npt.assert_equal(np.min(rs),0)

def test_rms_rmse():
    """

    Test the rms and rmse functions

    """

    # They both give the same answer:
    data = np.random.randn(10)
    npt.assert_equal(ozu.rms(data), ozu.rmse(data, np.zeros(data.shape)))
    
    
def test_xform():
    """
    Testing affine transformation of coordinates
    """

    coords = np.array([[1,2],[1,2],[1,2]])

    npt.assert_equal(ozu.xform(coords, np.eye(4)), coords)

    # Scaling by a factor of 2:
    aff = np.array([[2,0,0,0],
                    [0,2,0,0],
                    [0,0,2,0],
                    [0,0,0,1]])

    npt.assert_equal(ozu.xform(coords, aff), coords * 2) 

    # Translation by 1: 
    aff = np.array([[1,0,0,1],
                    [0,1,0,1],
                    [0,0,1,1],
                    [0,0,0,1]])

    npt.assert_equal(ozu.xform(coords, aff), coords + 1) 

    # Test error handling:

    coords = np.array([[1,2],[1,2]])

    npt.assert_raises(ValueError, ozu.xform, coords, aff)

    # Restore the sensible coordinates: 
    coords =  np.array([[1,2],[1,2],[1,2]])

    aff = np.eye(3)
    
    npt.assert_raises(ValueError, ozu.xform, coords, aff)
    
