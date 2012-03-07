import os
import tempfile

import numpy as np
import numpy.testing as npt

import nibabel as ni

import microtrack as mt
import microtrack.model as mtm
import microtrack.fibers as mtf
import microtrack.dwi as dwi
import microtrack.io as mio

# Initially, we want to check whether the data is available (would have to be
# downloaded separately, because it's huge): 
data_path = os.path.split(mt.__file__)[0] + '/data/'
if 'dwi.nii.gz' in os.listdir(data_path):
    no_data = False
else:
    no_data = True


# This takes some time, because it requires reading large data files and of
# course, needs to be skipped if the data is no where to be found:     
@npt.decorators.slow
@npt.decorators.skipif(no_data)
def test_BaseModel():
    
    BM = mtm.BaseModel(data_path + 'small_dwi.nii.gz',
                       data_path + 'dwi.bvecs',
                       data_path + 'dwi.bvals')

    npt.assert_equal(BM.r_squared, np.ones(BM.signal.shape[:3]))
    npt.assert_equal(BM.R_squared, np.ones(BM.signal.shape[:3]))
    npt.assert_equal(BM.coeff_of_determination, np.ones(BM.signal.shape[:3]))
    
    
@npt.decorators.slow
@npt.decorators.skipif(no_data)
def test_TensorModel():

    tensor_file = os.path.join(tempfile.gettempdir() + 'DTI.nii.gz')

    TM = mtm.TensorModel(data_path + 'small_dwi.nii.gz',
                         data_path + 'dwi.bvecs',
                         data_path + 'dwi.bvals',
                         tensor_file=tensor_file)
    
    # Make sure the shapes of things make sense: 
    npt.assert_equal(TM.model_params.shape, TM.data.shape[:3] + (12,))
    npt.assert_equal(TM.evals.shape, TM.data.shape[:3] + (3,))
    npt.assert_equal(TM.evecs.shape, TM.data.shape[:3] + (3,3))
    # Call the fit function to make sure it runs through smoothly:
    npt.assert_equal(TM.fit.shape, TM.signal.shape)
    

@npt.decorators.slow
@npt.decorators.skipif(no_data)
def test_FiberModel():
    """

    Test the initialization of FiberModel class instances
    
    """ 
    ad = 1.5
    rd = 0.5
    FG = mio.fg_from_pdb(data_path + 'FG_w_stats.pdb',
                     verbose=False)

    M = mtm.FiberModel(data_path + 'dwi.nii.gz',
                       data_path + 'dwi.bvecs',
                       data_path + 'dwi.bvals',
                       FG, ad, rd)

    npt.assert_equal(M.matrix.shape[0], M.flat_signal.shape[0])
    npt.assert_equal(M.matrix.shape[-1], len(FG.fibers))


@npt.decorators.slow
@npt.decorators.skipif(not 'CSD10.nii.gz' in os.listdir(data_path))
def test_SphericalHarmonicsModel():
    """
    Test the estimation of SH models.
    """

    model_coeffs = ni.load(data_path + 'CSD10.nii.gz').get_data()

    SHM = mtm.SphericalHarmonicsModel(data_path + 'dwi.nii.gz',
                                      data_path + 'dwi.bvecs',
                                      data_path + 'dwi.bvals',
                                      model_coeffs)
