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
