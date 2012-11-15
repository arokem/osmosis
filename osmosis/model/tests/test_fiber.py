import os

import numpy as np
import numpy.testing as npt

import osmosis as oz
import osmosis.io as mio
from osmosis.model.fiber import FiberModel

data_path = os.path.split(oz.__file__)[0] + '/data/'

@npt.decorators.slow
def test_FiberModel():
    """

    Test the initialization of FiberModel class instances
    
    """ 
    ad = 1.5
    rd = 0.5
    FG = mio.fg_from_pdb(data_path + 'FG_w_stats.pdb',
                     verbose=False)

    M = FiberModel(data_path + 'dwi.nii.gz',
                       data_path + 'dwi.bvecs',
                       data_path + 'dwi.bvals',
                       FG, ad, rd)

    npt.assert_equal(M.matrix[0].shape[0], np.prod(M.voxel_signal.shape))
    npt.assert_equal(M.matrix[0].shape[-1], len(FG.fibers))

    npt.assert_equal(M.matrix[1].shape[0], np.prod(M.voxel_signal.shape))
    npt.assert_equal(M.matrix[1].shape[-1], len(M.fg_idx_unique.T))
