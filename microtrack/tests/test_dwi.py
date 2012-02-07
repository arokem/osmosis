import numpy as np
import numpy.testing as npt
import nibabel as ni

import microtrack.dwi as dwi

# This takes some time, because it requires reading large data files
@npt.decorators.slow
def test_DWI():
    """
    Test the initialization of DWI class objects 
    """

    # XXX Need to figure out how to deal with this more elegantly:
    mrv_data_path = '/Users/arokem/source/vista/vistadata/'

    dwi_path = mrv_data_path + 'diffusion/dtiRawPreprocess/CNI/s1/33Vol/raw/'

    # Make one from strings: 
    D1 = dwi.DWI(dwi_path + 'dti.nii.gz',
            dwi_path + 'dti.bvec',
            dwi_path + 'dti.bval')

    # There should be agreement on the last dimension of each:
    npt.assert_equal(D1.data.shape[-1], 
                     D1.bvals.shape[-1])
    
    npt.assert_equal(D1.data.shape[-1],
                     D1.bvecs.shape[-1]) 

    # Make one from arrays: 
    data = ni.load(dwi_path + 'dti.nii.gz').get_data()
    bvecs = np.loadtxt(dwi_path + 'dti.bvec')
    bvals = np.loadtxt(dwi_path + 'dti.bval')

    D2 = dwi.DWI(data, bvecs, bvals)

    # It shouldn't matter:
    npt.assert_equal(D1.data, D2.data)
    npt.assert_equal(D1.bvecs, D2.bvecs)
    npt.assert_equal(D1.bvals, D2.bvals)

    npt.assert_equal(D1.affine.shape, (4,4))
    # When the data is provided as an array, there is no way to 
    npt.assert_equal(D2.affine, np.eye(4))
