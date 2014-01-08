import numpy as np
import numpy.testing as npt
import nibabel as nib
import os

import osmosis
import osmosis.multi_bvals as sfm_mb
import osmosis.model.sparse_deconvolution as sfm
import osmosis.mean_diffusivity_models as mdm
import osmosis.utils as ozu

data_path = os.path.join(osmosis.__path__[0], 'data')

data_pv = nib.load(os.path.join(data_path, "red_data.nii.gz")).get_data()
bvals_pv = np.loadtxt(os.path.join(data_path, "bvals"))
bvecs_pv = np.loadtxt(os.path.join(data_path, "bvecs"))

bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals_pv)
all_b_inds = np.where(rounded_bvals != 0)

mask_orig = nib.load(os.path.join(data_path, 'nodif_brain_mask.nii.gz')).get_data()
mask_pv = np.zeros(mask_orig.shape)
mask_pv[0, 0, 0:2] = 1

def test_optimize_MD_params():
    param_out, fit_out, ss_err = mdm.optimize_MD_params(data_pv, bvals_pv, mask_pv,
                                                        mdm.decaying_exp, -0.5)
    # Check to see if the right number of parameters are found
    npt.assert_equal(np.shape(param_out) == (2,1), 1)
    # Check to see if the sum of the squared errors is around the values that we want.
    npt.assert_equal(np.mean(ss_err) < 200, 1)
    
    param_out, fit_out, ss_err = mdm.optimize_MD_params(data_pv, bvals_pv, mask_pv,
                                                        mdm.decaying_exp_plus_const,
                                                        (-0.5, -0.5))
    npt.assert_equal(np.shape(param_out) == (2,2), 1)
    npt.assert_equal(np.mean(ss_err) < 200, 1)
    
    param_out, fit_out, ss_err = mdm.optimize_MD_params(data_pv, bvals_pv, mask_pv,
                                                        mdm.two_decaying_exp,
                                                        (-0.5, -0.5, -0.5))
    npt.assert_equal(np.shape(param_out) == (2,3), 1)
    npt.assert_equal(np.mean(ss_err) < 200, 1)
    
def test_kfold_xval_MD_mod():
    ss_err, predict_out = mdm.kfold_xval_MD_mod(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, mdm.decaying_exp,
                                                -0.5, 10)
    # Check to see if the sum of the squared errors is around the values that we want.
    npt.assert_equal(np.mean(ss_err) < 200, 1)
    
    ss_err, predict_out = mdm.kfold_xval_MD_mod(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, mdm.decaying_exp_plus_const,
                                                (-0.5, -0.5), 10)
    npt.assert_equal(np.mean(ss_err) < 200, 1)
    
    ss_err, predict_out = mdm.kfold_xval_MD_mod(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, mdm.two_decaying_exp,
                                                (-0.5, -0.5, -0.5), 10)
    npt.assert_equal(np.mean(ss_err) < 200, 1)