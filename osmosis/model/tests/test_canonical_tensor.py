import os
import tempfile

import numpy as np
import numpy.testing as npt

import nibabel as ni

import osmosis as oz
from osmosis.model.canonical_tensor import (CanonicalTensorModel,
                                            CanonicalTensorModelOpt)

data_path = os.path.split(oz.__file__)[0] + '/data/'


def test_CanonicalTensorModel():
    """

    Test the simple canonical + sphere model.

    """
    # 1000 'voxels' with constant data in each one in all directions (+b0): 
    data = (np.random.rand(10 * 10 * 10).reshape(10 * 10 * 10, 1) +
            np.zeros((10 * 10 * 10, 160))).reshape(10,10,10,160)

    CTM = CanonicalTensorModel(data,
                                   data_path + 'dwi.bvecs',
                                   data_path + 'dwi.bvals',
        params_file=tempfile.NamedTemporaryFile().name)
    
    # XXX Smoke testing only
    npt.assert_equal(CTM.fit.shape, CTM.signal.shape)

    mask_array = np.zeros(ni.load(data_path+'small_dwi.nii.gz').shape[:3])
    # Only two voxels:
    mask_array[1:3, 1:3, 1:3] = 1
    # Fit this on some real dwi data
    for mode in ['signal_attenuation', 'relative_signal', 'normalize', 'log']:
        for params_file in [None, tempfile.NamedTemporaryFile().name, 'temp']:
            CTM = CanonicalTensorModel(data_path+'small_dwi.nii.gz',
                                       data_path + 'dwi.bvecs',
                                       data_path + 'dwi.bvals',
                                       mask=mask_array,
                                       params_file=params_file,
                                       mode=mode)

            # XXX Smoke testing only:
            npt.assert_equal(CTM.fit.shape, CTM.signal.shape)
            npt.assert_equal(CTM.principal_diffusion_direction.shape,
                             CTM.signal.shape[:3] + (3,))
            npt.assert_equal(CTM.fractional_anisotropy.shape,
                             CTM.signal.shape[:3])
        # Test over-sampling:
        for over_sample in [362, 246]: # Over-sample from dipy and from
                                       # camino-points
            CTM = CanonicalTensorModel(data_path+'small_dwi.nii.gz',
                                           data_path + 'dwi.bvecs',
                                           data_path + 'dwi.bvals',
                                           mask=mask_array,
            params_file=tempfile.NamedTemporaryFile().name,
            over_sample=over_sample,
                mode=mode)

            # XXX Smoke testing only:
            npt.assert_equal(CTM.fit.shape, CTM.signal.shape)

            

    # This shouldn't be possible, because we don't have a sphere with 151
    # samples handy:
    npt.assert_raises(ValueError,
                      CanonicalTensorModel,
                      data_path+'small_dwi.nii.gz',
                      data_path + 'dwi.bvecs',
                      data_path + 'dwi.bvals',
                      **dict(mask=mask_array,
                           params_file=tempfile.NamedTemporaryFile().name,
                           over_sample=151))

    # If you provide an unrecognized mode, you get an error:
    npt.assert_raises(ValueError,
                      CanonicalTensorModel,
                      data_path+'small_dwi.nii.gz',
                      data_path + 'dwi.bvecs',
                      data_path + 'dwi.bvals',
                      **dict(mask=mask_array,
                             mode='crazy_mode',
                             params_file=tempfile.NamedTemporaryFile().name))

                      
def test_CanonicalTensorModelOpt():
    """
    Test fitting of the CanonicalTensorModel by optimization
    """
    
    mask_array = np.zeros(ni.load(data_path+'small_dwi.nii.gz').shape[:3])
    # Only two voxels:
    mask_array[1:3, 1:3, 1:3] = 1

    # Fit this on some real dwi data
    for model_form in ['flexible', 'constrained', 'ball_and_stick']:
        for mode in ['relative_signal', 'signal_attenuation']:
            CTM = CanonicalTensorModelOpt(data_path+'small_dwi.nii.gz',
                                              data_path + 'dwi.bvecs',
                                              data_path + 'dwi.bvals',
                                              model_form = model_form,
                                              mode = mode,
                                              mask=mask_array,
                params_file=tempfile.NamedTemporaryFile().name)
        
        # XXX Smoke testing for now:
        npt.assert_equal(CTM.fit.shape, CTM.signal.shape)

    # Normalize doesn't make sense for the optimization, so we raise an error
    npt.assert_raises(ValueError,
                     CanonicalTensorModelOpt,
                     data_path+'small_dwi.nii.gz',
                     data_path + 'dwi.bvecs',
                     data_path + 'dwi.bvals',
                     mode='normalize',
                     mask=mask_array,
                     params_file=tempfile.NamedTemporaryFile().name)

    npt.assert_raises(ValueError,
                      CanonicalTensorModelOpt,
                      data_path+'small_dwi.nii.gz',
                      data_path + 'dwi.bvecs',
                      data_path + 'dwi.bvals',
                      model_form='crazy_model',
                      mode='normalize',
                      mask=mask_array,
                      params_file=tempfile.NamedTemporaryFile().name)
