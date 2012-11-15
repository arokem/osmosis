import os

import numpy as np
import numpy.testing as npt
import scipy.stats as stats

import nibabel as ni


import osmosis as oz
import osmosis.model.base as ozm
import osmosis.utils as ozu
from osmosis.model.base import DWI, BaseModel

data_path = os.path.split(oz.__file__)[0] + '/data/'

@npt.decorators.slow
def test_DWI():
    """
    Test the initialization of DWI class objects 
    """
    
    # Make one from strings: 
    D1 = DWI(data_path + 'small_dwi.nii.gz',
            data_path + 'dwi.bvecs',
            data_path + 'dwi.bvals')

    # There should be agreement on the last dimension of each:
    npt.assert_equal(D1.data.shape[-1], 
                     D1.bvals.shape[-1])
    
    npt.assert_equal(D1.data.shape[-1],
                     D1.bvecs.shape[-1]) 

    npt.assert_equal(D1.affine.shape, (4,4))

    # Provide the affine as input:
    D1_w_affine = DWI(data_path + 'small_dwi.nii.gz',
            data_path + 'dwi.bvecs',
            data_path + 'dwi.bvals',
            affine=ni.load(data_path + 'small_dwi.nii.gz').get_affine())

    npt.assert_equal(D1_w_affine.affine, D1.affine)

    # Provide a mask as input:
    D1_w_mask = DWI(data_path + 'small_dwi.nii.gz',
            data_path + 'dwi.bvecs',
            data_path + 'dwi.bvals',
            mask = data_path + 'small_dwi_mask.nii.gz')

    npt.assert_equal(D1_w_mask.shape, D1.shape)

    # Try sub-sampling only 30 directions:
    D1_subsampled = DWI(data_path + 'small_dwi.nii.gz',
                            data_path + 'dwi.bvecs',
                            data_path + 'dwi.bvals',
                            sub_sample=30)

    # There should now be 30 non-zero b values
    npt.assert_equal(D1_subsampled.b_idx.shape[0], 30)
    # There are still 10 b0 volumes:
    npt.assert_equal(D1_subsampled.bvecs.shape[-1], 40)
    npt.assert_equal(D1_subsampled.bvals.shape[0], 40)

    # Make one from arrays: 
    data = ni.load(data_path + 'small_dwi.nii.gz').get_data()
    bvecs = np.loadtxt(data_path + 'dwi.bvecs')
    bvals = np.loadtxt(data_path + 'dwi.bvals')

    D2 = DWI(data, bvecs, bvals)

    # It shouldn't matter:
    npt.assert_equal(D1.data, D2.data)
    npt.assert_equal(D1.bvecs, D2.bvecs)
    npt.assert_equal(D1.bvals, D2.bvals)

    # Should be possible to sub-sample using specific bvecs:
    D2_sub_sampled = DWI(data, bvecs,bvals, sub_sample=np.array([0,1,2]))

    # You should still have all the b0 vectors there: 
    npt.assert_equal(D2_sub_sampled.bvecs[:,10:12], bvecs[:,10:12])
    
    # When the data is provided as an array, there is no way to know what the
    # affine is, so we throw a warning, and set it to np.eye(4):

    # XXX auto-attr probably makes calling this tricky:
    # npt.assert_warns(exceptions.UserWarning, D2.affine)
    
    npt.assert_equal(D2.affine, np.eye(4))

    npt.assert_equal(D2.shape, data.shape)

    # If the data is neither an array, nor a file-name, that should raise an
    # error: 
    npt.assert_raises(ValueError, DWI, [1,2,3], bvecs, bvals)

    # Test signal reliability.
    # Default behavior is Pearson correlation:
    npt.assert_equal(D1.signal_reliability(D2),
                     np.ones(D1.shape[:3]))
    # Optionally, we can use linregress and ask for the r_idx to be 2:
    npt.assert_equal(D1.signal_reliability(D2,
                    correlator=stats.linregress, r_idx=2), np.ones(D1.shape[:3]))

    # Or ask for coeff_of_determination, where there is only one output
    # (r_idx=np.nan) and no need to squre it to get the reliability measure:
    npt.assert_equal(D1.signal_reliability(D2,
                                           correlator=ozu.coeff_of_determination,
                                           r_idx=np.nan,
                                           square=False),
                      np.ones(D1.shape[:3]))

    # This should all work even if numexpr is not present, so let's test that
    # as well, in systems where numexpr is present: 
    has_numexpr = ozm.has_numexpr
    if has_numexpr:  #XXX Doesn't seem to do that for now
        ozm.has_numexpr = False
        D1 = DWI(data_path + 'small_dwi.nii.gz',
                     data_path + 'dwi.bvecs',
                     data_path + 'dwi.bvals')
        D2 = DWI(data, bvecs, bvals)

        npt.assert_equal(D1.signal_reliability(D2),
                     np.ones(D1.shape[:3]))
        
        npt.assert_equal(D1.signal_reliability(D2,
                                               correlator=stats.linregress,
                                               r_idx=2),
                         np.ones(D1.shape[:3]))

        npt.assert_equal(D1.signal_reliability(D2,
                                           correlator=ozu.coeff_of_determination,
                                           r_idx=np.nan,
                                           square=False),
                      np.ones(D1.shape[:3]))
        # Set it back:
        ozm.has_numexpr = True

    npt.assert_equal(D1.signal_attenuation, 1-D1.relative_signal)


@npt.decorators.slow
def test_BaseModel():
    
    BM = BaseModel(data_path + 'small_dwi.nii.gz',
                       data_path + 'dwi.bvecs',
                       data_path + 'dwi.bvals')

    npt.assert_equal(BM.r_squared, np.ones(BM.signal.shape[:3]))
    npt.assert_equal(BM.R_squared, np.ones(BM.signal.shape[:3]))
    npt.assert_equal(BM.coeff_of_determination, np.ones(BM.signal.shape[:3]))
    npt.assert_equal(BM.RMSE, np.zeros(BM.signal.shape[:3]))

    # Test without numexpr: XXX Doesn't seem to do that for now
    has_numexpr = ozm.has_numexpr
    if has_numexpr:
        ozm.has_numexpr = False
        BM_wo_numexpr = BaseModel(data_path + 'small_dwi.nii.gz',
                       data_path + 'dwi.bvecs',
                       data_path + 'dwi.bvals')

        npt.assert_equal(BM.r_squared, np.ones(BM.signal.shape[:3]))
        npt.assert_equal(BM.R_squared, np.ones(BM.signal.shape[:3]))
        npt.assert_equal(BM.coeff_of_determination, np.ones(BM.signal.shape[:3]))
        npt.assert_equal(BM.RMSE, np.zeros(BM.signal.shape[:3]))

        # Set it back:
        ozm.has_numexpr = True

