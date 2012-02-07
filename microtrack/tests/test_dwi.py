import os
import exceptions

import numpy as np
import numpy.testing as npt
import nibabel as ni

import microtrack as mt
import microtrack.dwi as dwi

# Initially, we want to check whether the data is available (would have to be
# downloaded separately, because it's huge): 
data_path = os.path.split(mt.__file__)[0] + '/data/'
if 'dwi.nii.gz' in os.listdir(data_path):
    no_data = False
else:
    no_data = True


# This takes some time, because it requires reading large data files
@npt.decorators.slow
@npt.decorators.skipif(no_data)
def test_DWI():
    """
    Test the initialization of DWI class objects 
    """
    
    
    # Make one from strings: 
    D1 = dwi.DWI(data_path + 'dwi.nii.gz',
            data_path + 'dwi.bvecs',
            data_path + 'dwi.bvals')

    # There should be agreement on the last dimension of each:
    npt.assert_equal(D1.data.shape[-1], 
                     D1.bvals.shape[-1])
    
    npt.assert_equal(D1.data.shape[-1],
                     D1.bvecs.shape[-1]) 

    # Make one from arrays: 
    data = ni.load(data_path + 'dwi.nii.gz').get_data()
    bvecs = np.loadtxt(data_path + 'dwi.bvecs')
    bvals = np.loadtxt(data_path + 'dwi.bvals')

    D2 = dwi.DWI(data, bvecs, bvals)

    # It shouldn't matter:
    npt.assert_equal(D1.data, D2.data)
    npt.assert_equal(D1.bvecs, D2.bvecs)
    npt.assert_equal(D1.bvals, D2.bvals)

    npt.assert_equal(D1.affine.shape, (4,4))
    # When the data is provided as an array, there is no way to know what the
    # affine is, so we throw a warning, and set it to np.eye(4):

    # XXX auto-attr probably makes calling this tricky:
    # npt.assert_warns(exceptions.UserWarning, D2.affine)
    
    npt.assert_equal(D2.affine, np.eye(4))
