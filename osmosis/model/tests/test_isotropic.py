import numpy as np
import numpy.testing as npt
import nibabel as nib
import os

import osmosis
import osmosis.model.sparse_deconvolution as sfm
import osmosis.model.isotropic as mdm
import osmosis.utils as ozu

data_path = os.path.join(osmosis.__path__[0], 'data')

data_pv = nib.load(os.path.join(data_path, "red_data.nii.gz")).get_data()
bvals_pv = np.loadtxt(os.path.join(data_path, "bvals"))
bvecs_pv = np.loadtxt(os.path.join(data_path, "bvecs"))

bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals_pv)
all_b_inds = np.where(rounded_bvals != 0)

mask_pv = np.zeros(data_pv.shape[:3])
mask_pv[0, 0, 0:2] = 1

def test_isotropic_params():
    param_out, fit_out, ss_err = mdm.isotropic_params(data_pv, bvals_pv,
                                                        bvecs_pv, mask_pv,
                                                        "decaying_exp",
                                                        initial = -0.5,
                                                        bounds = [(None, None)],
                                                        signal="log")
    # Check to see if the right number of parameters are found
    npt.assert_equal(np.shape(param_out) == (2,1), 1)
    # Check to see if the sum of the squared errors is around the values
    # that we want.
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    param_out, fit_out, ss_err = mdm.isotropic_params(data_pv, bvals_pv, bvecs_pv,
                                                        mask_pv,
                                                        "decaying_exp_plus_const",
                                                        initial=(-0.5, -0.5),
                                                        bounds=[(None, None),
                                                                (None, None)],
                                                        signal="log")
    npt.assert_equal(np.shape(param_out) == (2,2), 1)
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    param_out, fit_out, ss_err = mdm.isotropic_params(data_pv, bvals_pv,
                                                        bvecs_pv, mask_pv,
                                                        "two_decaying_exp",
                                                        initial=(-0.5, -0.5, -0.5),
                                                        bounds=[(None, None),
                                                                (None, None),
                                                                (None, None)],
                                                        signal="log")
    npt.assert_equal(np.shape(param_out) == (2,3), 1)
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    param_out, fit_out, ss_err = mdm.isotropic_params(data_pv, bvals_pv, bvecs_pv,
                                                        mask_pv, "single_exp_rs",
                                                        signal="relative_signal")
    npt.assert_equal(np.shape(param_out) == (2,1), 1)
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    param_out, fit_out, ss_err = mdm.isotropic_params(data_pv, bvals_pv, bvecs_pv,
                                                        mask_pv,
                                                        "single_exp_nf_rs",
                                                        signal="relative_signal")
    npt.assert_equal(np.shape(param_out) == (2,2), 1)
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    param_out, fit_out, ss_err = mdm.isotropic_params(data_pv, bvals_pv, bvecs_pv,
                                                        mask_pv, "bi_exp_rs",
                                                        signal="relative_signal")
    npt.assert_equal(np.shape(param_out) == (2,3), 1)
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    param_out, fit_out, ss_err = mdm.isotropic_params(data_pv, bvals_pv, bvecs_pv,
                                                        mask_pv, "bi_exp_nf_rs",
                                                        signal="relative_signal")
    npt.assert_equal(np.shape(param_out) == (2,4), 1)
    npt.assert_equal(np.mean(ss_err) < 200, 1)

def test_kfold_xval_MD_mod():
    ss_err, predict_out = mdm.kfold_xval_MD_mod(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, "decaying_exp",
                                                10, initial = -0.5,
                                                bounds=[(None, None)],
                                                signal="log")
    # Check to see if the sum of the squared errors is around the values
    # that we want.
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    ss_err, predict_out = mdm.kfold_xval_MD_mod(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, "decaying_exp_plus_const",
                                                10, initial = (-0.5, -0.5),
                                                bounds=[(None, None), (None, None)],
                                                signal="log")
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    ss_err, predict_out = mdm.kfold_xval_MD_mod(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, "two_decaying_exp",
                                                10, initial=(-0.5, -0.5, -0.5),
                                                bounds=[(None, None), (None, None),
                                                (None, None)],
                                                signal="log")
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    ss_err, predict_out = mdm.kfold_xval_MD_mod(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, "single_exp_rs", 10)
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    ss_err, predict_out = mdm.kfold_xval_MD_mod(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, "single_exp_nf_rs", 10)
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    ss_err, predict_out = mdm.kfold_xval_MD_mod(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, "bi_exp_rs", 10)
    npt.assert_equal(np.mean(ss_err) < 200, 1)

    ss_err, predict_out = mdm.kfold_xval_MD_mod(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, "bi_exp_nf_rs", 10)
    npt.assert_equal(np.mean(ss_err) < 200, 1)
