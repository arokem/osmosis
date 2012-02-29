import os
import numpy.testing as npt

import nibabel as ni

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
    # Smoke testing this: 
    fg = mtv.nii2fg(pdb_file, nii_file)

    # XXX Need to come up with more rigorous tests here

def test_fg2volume():

    data_path = os.path.split(mt.__file__)[0] + '/data/'
    pdb_file = data_path + 'FG_w_stats.pdb'
    nii_file = data_path + 'fp20110912_ecc.nii.gz'
    fg = mtv.nii2fg(pdb_file, nii_file)
    # Smoke testing this: 
    vol = mtv.fg2volume(fg, 'fp20110912_ecc.nii.gz',
                        shape=ni.load(nii_file).get_shape())
