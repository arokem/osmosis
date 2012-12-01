import os
import numpy as np
import numpy.testing as npt

import nibabel as ni

import osmosis as oz
import osmosis.volume as ozv
import osmosis.io as oio
import osmosis.fibers as ozf

def test_nii2fg():
    """

    Testing the addition of information from a nifti volume to fiber groups

    nii2fg

    """

    data_path = os.path.split(oz.__file__)[0] + '/data/'
    pdb_file = data_path + 'FG_w_stats.pdb'
    nii_file = data_path + 'fp20110912_ecc.nii.gz'
    # Smoke testing this: 
    fg = ozv.nii2fg(oio.fg_from_pdb(pdb_file), nii_file)

    # XXX Need to come up with more rigorous tests here

def test_fg2volume():

    data_path = os.path.split(oz.__file__)[0] + '/data/'
    pdb_file = data_path + 'FG_w_stats.pdb'
    nii_file = data_path + 'fp20110912_ecc.nii.gz'
    fg = ozv.nii2fg(oio.fg_from_pdb(pdb_file), nii_file)
    # Smoke testing this: 
    vol = ozv.fg2volume(fg, 'fp20110912_ecc.nii.gz',
                        shape=ni.load(nii_file).get_shape())

def test_resample_volume():
    """
    Testing resampling of one volume into another volumes space (a t1 into a
    dwi space, for example) 
    """

    source = ni.Nifti1Image(np.random.randn(10,10,10), np.eye(4))
    target = ni.Nifti1Image(np.random.randn(10,10,10), np.eye(4))
    new_vol = ozv.resample_volume(source, target)

    npt.assert_equal(new_vol.shape, target.shape[:3])
    
    
