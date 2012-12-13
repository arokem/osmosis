import os
import tempfile

import numpy as np
import numpy.testing as npt

import nibabel as ni

import osmosis as oz
from osmosis.model.sparse_deconvolution import SparseDeconvolutionModel

data_path = os.path.split(oz.__file__)[0] + '/data/'


def test_SparseDeconvolutionModel():
    """

    Test the sparse deconvolution model.

    """
    data = (np.random.rand(10 * 10 * 10).reshape(10 * 10 * 10, 1) +
            np.zeros((10 * 10 * 10, 160))).reshape(10, 10, 10,160)

    mask_array = np.zeros(ni.load(data_path+'small_dwi.nii.gz').shape[:3])
    # Only two voxels:
    mask_array[1:3, 1:3, 1:3] = 1

    SSD = SparseDeconvolutionModel(data,
                                   data_path + 'dwi.bvecs',
                                   data_path + 'dwi.bvals',
                                   mask=mask_array,
        params_file=tempfile.NamedTemporaryFile().name)
    
    # XXX Smoke testing only
    npt.assert_equal(SSD.fit.shape, SSD.signal.shape)

    # Fit this on some real dwi data
    for mode in ['signal_attenuation', 'relative_signal', 'normalize', 'log']:
        for params_file in [None, tempfile.NamedTemporaryFile().name, 'temp']:
            SSD = SparseDeconvolutionModel(data_path+'small_dwi.nii.gz',
                                       data_path + 'dwi.bvecs',
                                       data_path + 'dwi.bvals',
                                       mask=mask_array,
                                       params_file=params_file,
                                       mode=mode)

            # XXX Smoke testing only:
            npt.assert_equal(SSD.fit.shape, SSD.signal.shape)
            npt.assert_equal(SSD.principal_diffusion_direction.shape,
                             SSD.signal.shape + (3,))
            npt.assert_equal(SSD.fractional_anisotropy.shape,
                             SSD.signal.shape[:3])
        # Test over-sampling:
        for over_sample in [362, 246]: # Over-sample from dipy and from
                                       # camino-points
            SSD = SparseDeconvolutionModel(data_path+'small_dwi.nii.gz',
                                           data_path + 'dwi.bvecs',
                                           data_path + 'dwi.bvals',
                                           mask=mask_array,
            params_file=tempfile.NamedTemporaryFile().name,
            over_sample=over_sample,
                mode=mode)

            # XXX Smoke testing only:
            npt.assert_equal(SSD.fit.shape, SSD.signal.shape)

            

    # This shouldn't be possible, because we don't have a sphere with 151
    # samples handy:
    npt.assert_raises(ValueError,
                      SparseDeconvolutionModel,
                      data_path+'small_dwi.nii.gz',
                      data_path + 'dwi.bvecs',
                      data_path + 'dwi.bvals',
                      **dict(mask=mask_array,
                           params_file=tempfile.NamedTemporaryFile().name,
                           over_sample=151))

    # If you provide an unrecognized mode, you get an error:
    npt.assert_raises(ValueError,
                      SparseDeconvolutionModel,
                      data_path+'small_dwi.nii.gz',
                      data_path + 'dwi.bvecs',
                      data_path + 'dwi.bvals',
                      **dict(mask=mask_array,
                             mode='crazy_mode',
                             params_file=tempfile.NamedTemporaryFile().name))

                      

def test_predict():
    """
    Test the SparseDeconvolutionModel predict method
    """
    data = (np.random.rand(10 * 10 * 10).reshape(10 * 10 * 10, 1) +
            np.zeros((10 * 10 * 10, 160))).reshape(10, 10, 10,160)

    mask_array = np.zeros(ni.load(data_path+'small_dwi.nii.gz').shape[:3])
    # Only two voxels:
    mask_array[1:3, 1:3, 1:3] = 1

    SSD = SparseDeconvolutionModel(data,
                                   data_path + 'dwi.bvecs',
                                   data_path + 'dwi.bvals',
                                   mask=mask_array,
        params_file=tempfile.NamedTemporaryFile().name)

    bvecs = SSD.bvecs[:, SSD.b_idx]
    new_bvecs = bvecs[:,:4]
    prediction = SSD.predict(new_bvecs)
    npt.assert_array_equal(prediction, SSD.fit[...,:4])
