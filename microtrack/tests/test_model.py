import os
import tempfile

import numpy as np
import numpy.testing as npt

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
    
    DWI = dwi.DWI(data_path + 'small_dwi.nii.gz',
                  data_path + 'dwi.bvecs',
                  data_path + 'dwi.bvals')

    BM = mtm.BaseModel(DWI)
    npt.assert_equal(BM.r_squared, np.ones(BM.signal.shape[:3]))
    
@npt.decorators.slow
@npt.decorators.skipif(no_data)
def test_TensorModel():

    DWI = dwi.DWI(data_path + 'small_dwi.nii.gz',
                  data_path + 'dwi.bvecs',
                  data_path + 'dwi.bvals')

    file_name = os.path.join(tempfile.gettempdir() + 'DTI.nii.gz')

    TM = mtm.TensorModel(DWI,file_name=file_name)
    
    # Make sure the shapes of things make sense: 
    npt.assert_equal(TM.model_params.shape, DWI.data.shape[:3] + (12,))
    npt.assert_equal(TM.evals.shape, DWI.data.shape[:3] + (3,))
    npt.assert_equal(TM.evecs.shape, DWI.data.shape[:3] + (3,3))

    

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

    DWI = dwi.DWI(data_path + 'dwi.nii.gz',
              data_path + 'dwi.bvecs',
              data_path + 'dwi.bvals')

    M = mtm.FiberModel(DWI, FG, ad, rd)

    npt.assert_equal(M.matrix.shape[0], M.flat_signal.shape[0])
    npt.assert_equal(M.matrix.shape[-1], len(FG.fibers))

def test_SphericalHarmonicsModel():

    pass 
