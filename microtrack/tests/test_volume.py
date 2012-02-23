import os
import numpy.testing as npt

import microtrack as mt
import microtrack.volume as mtv
import microtrack.fibers as mtf

def test_nii2fg():
    """

    Testing the addition of information from a nifti volume to fiber groups

    """

    data_path = os.path.split(mt.__file__)[0] + '/data/'
    pdb_file = data_path + 'FG_w_stats.pdb'
    nii_file = data_path + 'fp20110912_ecc.nii.gz'
    fg = mtv.nii2fg(pdb_file, nii_file)
