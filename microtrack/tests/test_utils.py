import numpy as np
import numpy.testing as npt

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
    
    

    
    

