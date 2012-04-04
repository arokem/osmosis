import os
import numpy.testing as npt

import nibabel as ni

import osmosis as oz
import osmosis.volume as ozv
import osmosis.fibers as ozf

def test_nii2fg():
    """

    Testing the addition of information from a nifti volume to fiber groups

    """

    data_path = os.path.split(oz.__file__)[0] + '/data/'
    pdb_file = data_path + 'FG_w_stats.pdb'
    nii_file = data_path + 'fp20110912_ecc.nii.gz'
    # Smoke testing this: 
    fg = ozv.nii2fg(pdb_file, nii_file)

    # XXX Need to come up with more rigorous tests here

def test_fg2volume():

    data_path = os.path.split(oz.__file__)[0] + '/data/'
    pdb_file = data_path + 'FG_w_stats.pdb'
    nii_file = data_path + 'fp20110912_ecc.nii.gz'
    fg = ozv.nii2fg(pdb_file, nii_file)
    # Smoke testing this: 
    vol = ozv.fg2volume(fg, 'fp20110912_ecc.nii.gz',
                        shape=ni.load(nii_file).get_shape())

def test_resample_volume():
    """
    Testing resampling of a t1 into a dwi space (and such...)
    """
    
    data_path = os.path.split(oz.__file__)[0] + '/data/'
    source_file = data_path + 'FP_t1.nii.gz'
    target_file = (data_path +
                   '0005_01_DTI_2mm_150dir_2x_b2000_aligned_trilin.nii.gz')

    new_vol = ozv.resample_volume(source_file, target_file)

    npt.assert_equal(new_vol.shape, ni.load(target_file).shape[:3])
    
    
