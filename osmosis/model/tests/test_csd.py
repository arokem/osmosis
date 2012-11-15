import os

import numpy as np
import numpy.testing as npt

import nibabel as ni
import osmosis as oz
from osmosis.model.csd import SphericalHarmonicsModel
import osmosis.model.analysis as ma
data_path = os.path.split(oz.__file__)[0] + '/data/'

@npt.decorators.slow
@npt.decorators.skipif(not 'CSD10.nii.gz' in os.listdir(data_path))
def test_SphericalHarmonicsModel():
    """
    Test the estimation of SH models.
    """
    
    model_coeffs = ni.load(data_path + 'CSD10.nii.gz').get_data()

    mask = np.zeros(model_coeffs.shape[:3])
    # Do this in only one voxel:
    mask[40, 40, 40] = 1
    response_file = (data_path +
                     '0009_01_DWI_2mm150dir_2x_b1000_aligned_trilin_ER.mif')
    
    for response in [response_file, None]:
        SHM1 = SphericalHarmonicsModel(data_path + 'dwi.nii.gz',
                                      data_path + 'dwi.bvecs',
                                      data_path + 'dwi.bvals',
                                      model_coeffs,
                                      response_file = response,
                                      mask=mask)

        # Can also provide the input as a string:
        model_coeffs = data_path + 'CSD10.nii.gz'

        SHM2 = SphericalHarmonicsModel(data_path + 'dwi.nii.gz',
                                          data_path + 'dwi.bvecs',
                                          data_path + 'dwi.bvals',
                                          model_coeffs,
                                          response_file = response,
                                          mask=mask)

        # Smoke testing:
        SHM1.fit
        SHM1.odf_peaks
        SHM1.crossing_index
        pdd_reliab = ma.pdd_reliability(SHM1, SHM2) 
        npt.assert_almost_equal(pdd_reliab[40, 40, 40], 0)
        
    # Check error-handling
    # In this one, the input is not a proper array (why I am testing this
    # here?): 
    npt.assert_raises(ValueError,
                      SphericalHarmonicsModel,
                      [1,2,3],
                      data_path + 'dwi.bvecs',
                      data_path + 'dwi.bvals',
                      model_coeffs,
                      None,None,None,
                      response_file)

    # In this one, both a response function coefficients file and AD/RD for the
    # calculation of a tensor are provided:
    npt.assert_raises(ValueError,
                      SphericalHarmonicsModel,
                      data_path + 'dwi.nii.gz',
                      data_path + 'dwi.bvecs',
                      data_path + 'dwi.bvals',
                      model_coeffs,
                      None, 1.5, 0.5,
                      response_file)
